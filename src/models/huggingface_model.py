"""HuggingFace model implementation."""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import logging
import re
import os
import time

from .base import BaseModel

# Monkey-patch для Phi-4: LossKwargs добавлен в transformers 4.50+, в старых версиях отсутствует
def _patch_transformers_for_phi4():
    """Добавить LossKwargs в transformers.utils если отсутствует (для Phi-4)."""
    import transformers.utils as tf_utils
    if not hasattr(tf_utils, 'LossKwargs'):
        from typing import TypedDict
        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: int
        tf_utils.LossKwargs = LossKwargs
_patch_transformers_for_phi4()

# Monkey-patch для совместимости MiniCPM4 с новыми версиями transformers
# Модель MiniCPM4 импортирует is_torch_fx_available, удалённый в transformers 5.x
def _patch_transformers_for_minicpm():
    """Добавить is_torch_fx_available в import_utils если отсутствует."""
    from transformers.utils import import_utils
    if not hasattr(import_utils, 'is_torch_fx_available'):
        def is_torch_fx_available():
            # Возвращаем False — блок torch.fx.wrap не нужен для инференса,
            # это избегает зависимости от is_torch_greater_or_equal_than_1_13
            return False
        import_utils.is_torch_fx_available = is_torch_fx_available
_patch_transformers_for_minicpm()

# Try to import huggingface_hub for download progress tracking
try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import tqdm as hf_tqdm
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class DownloadProgressCallback:
    """Callback для отслеживания прогресса загрузки файлов модели."""
    
    def __init__(self, logger):
        self.logger = logger
        self.current_file = None
        self.file_start_time = None
        self.last_log_time = 0
        self.log_interval = 2.0  # Логировать каждые 2 секунды
    
    def __call__(self, chunk_size, total_size=None):
        """Вызывается при загрузке каждого чанка."""
        import time
        current_time = time.time()
        
        # Логируем прогресс только периодически, чтобы не засорять логи
        if current_time - self.last_log_time >= self.log_interval:
            if total_size:
                downloaded_mb = (chunk_size * 100) / (1024 * 1024) if chunk_size else 0
                total_mb = total_size / (1024 * 1024)
                percent = (chunk_size / total_size * 100) if total_size > 0 else 0
                self.logger.info(f"   📥 Загружено: {downloaded_mb:.1f} MB / {total_mb:.1f} MB ({percent:.1f}%)")
            else:
                downloaded_mb = chunk_size / (1024 * 1024) if chunk_size else 0
                self.logger.info(f"   📥 Загружено: {downloaded_mb:.1f} MB")
            
            self.last_log_time = current_time


# Monkey-patch для исправления проблемы совместимости torch.is_autocast_enabled()
# Некоторые модели (например, Nanbeige) вызывают torch.is_autocast_enabled(device),
# но в PyTorch 2.2+ эта функция не принимает аргументов
_original_is_autocast_enabled = torch.is_autocast_enabled
def _patched_is_autocast_enabled(*args, **kwargs):
    """Обертка для torch.is_autocast_enabled, игнорирующая аргументы."""
    return _original_is_autocast_enabled()
torch.is_autocast_enabled = _patched_is_autocast_enabled


class HuggingFaceModel(BaseModel):
    """HuggingFace transformer model implementation."""
    
    def __init__(self, config):
        """Initialize HuggingFace model.
        
        Args:
            config: Model configuration containing:
                - model_path: HuggingFace model name or path
                - max_length: Maximum sequence length
                - temperature, top_p, top_k, repetition_penalty, max_new_tokens: см. generate (игнорируются, если use_hf_generation_config=true)
                - use_hf_generation_config: брать эти параметры из generation_config.json репозитория после загрузки весов
                - device: Device to use (cuda/cpu)
                - use_flash_attention: Whether to use flash attention
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.last_prompt = None  # Store last used prompt for logging
        self.model_path = config.get('model_path', 'gpt2')
        self.use_hf_generation_config = config.get('use_hf_generation_config', False)
        
        # Extract config parameters (max_length set after tokenizer load to use model's limit)
        self._config_max_length = config.get('max_length')
        # Параметры генерации: либо из YAML (по умолчанию), либо из generation_config.json модели
        if self.use_hf_generation_config:
            self._do_sample = None  # Задаётся из HF generation_config после загрузки весов
            self.top_k = None
            self.temperature = 0.7
            self.top_p = 0.9
            self.repetition_penalty = 1.1
            self.max_new_tokens = 150
        else:
            self._do_sample = None
            self.top_k = config.get('top_k')  # Опционально из YAML
            self.temperature = config.get('temperature', 0.7)
            self.top_p = config.get('top_p', 0.9)
            self.repetition_penalty = config.get('repetition_penalty', 1.1)
            self.max_new_tokens = config.get('max_new_tokens', 150)
        
        # Device configuration
        device_config = config.get('device', 'cuda')
        if device_config == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"🎯 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("💻 Using CPU")
        
        # Load model and tokenizer
        self.logger.info(f"📦 Начало загрузки модели: {self.model_path}")
        
        try:
            # Check if model is cached
            cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or os.path.expanduser('~/.cache/huggingface')
            self.logger.info(f"💾 Кэш моделей: {cache_dir}")
            
            # Получаем токен HuggingFace из конфига или переменной окружения
            hf_token = config.get('hf_token') or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
            if hf_token:
                self.logger.info("🔑 Используется HuggingFace token для доступа к модели")
            
            token_kwargs = {'trust_remote_code': True}
            if hf_token:
                token_kwargs['token'] = hf_token
            
            # Load tokenizer
            self.logger.info("📥 Загрузка токенайзера...")
            tokenizer_start_time = time.time()
            # Используем use_fast=False для совместимости со старыми версиями tokenizers
            # Если возникает ошибка ModelWrapper, это поможет использовать медленный токенайзер
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    use_fast=True,
                    **token_kwargs
                )
                self.logger.info("✅ Токенайзер загружен (быстрый режим)")
            except Exception as e:
                self.logger.warning(f"⚠️  Не удалось загрузить быстрый токенайзер: {e}")
                self.logger.info("🔄 Пробуем использовать медленный токенайзер...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    use_fast=False,
                    **token_kwargs
                )
                self.logger.info("✅ Токенайзер загружен (медленный режим)")
            
            tokenizer_time = time.time() - tokenizer_start_time
            self.logger.info(f"⏱️  Время загрузки токенайзера: {tokenizer_time:.2f} сек")
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # max_length: prefer model's context limit (tokenizer.model_max_length), fallback to config
            # tokenizer.model_max_length can be int(1e30) when not defined — treat as "no limit"
            model_max = getattr(self.tokenizer, 'model_max_length', None)
            if model_max is None or model_max >= 1_000_000_000:
                model_max = None
            self.max_length = model_max or self._config_max_length or 512
            self.logger.info(f"📏 max_length: {self.max_length} (model={model_max}, config={self._config_max_length})")
            
            # Determine dtype: use model's preferred format (bf16/fp16) if GPU supports it
            torch_dtype = self._get_torch_dtype()
            
            # Load model
            self.logger.info("📥 Загрузка модели (это может занять некоторое время)...")
            model_start_time = time.time()
            
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch_dtype,
            }
            if hf_token:
                model_kwargs['token'] = hf_token
            
            # Add flash attention if requested and available
            if config.get('use_flash_attention', False):
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                self.logger.info("⚡ Используется Flash Attention 2")
            
            # Log download progress
            self.logger.info("📥 Загрузка файлов модели из Hugging Face Hub...")
            
            # Включаем прогресс-бары Hugging Face Hub
            # Они будут выводиться в stderr и перехватываться run_batch_experiments.py
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
            
            # Определяем путь к кэшу модели для отслеживания прогресса
            # HuggingFace хранит модели в hub/models--org--name (новый формат)
            # или в models--org--name в корне (старый формат)
            from pathlib import Path
            model_cache_name = self.model_path.replace('/', '--')
            model_cache_dir = f'models--{model_cache_name}'
            model_cache_path = Path(cache_dir) / 'hub' / model_cache_dir
            if not model_cache_path.exists():
                model_cache_path = Path(cache_dir) / model_cache_dir
            
            # Проверяем, есть ли модель в кэше
            if model_cache_path.exists():
                self.logger.info(f"   ✅ Модель найдена в кэше: {model_cache_path}")
                # Подсчитываем размер файлов в кэше
                total_size = sum(f.stat().st_size for f in model_cache_path.rglob('*') if f.is_file())
                total_size_gb = total_size / (1024 ** 3)
                self.logger.info(f"   💾 Размер кэшированных файлов: {total_size_gb:.2f} GB")
            else:
                self.logger.info(f"   ⏳ Модель не найдена в кэше, начинается загрузка...")
                self.logger.info(f"   📥 Прогресс загрузки будет отображаться в логах ниже (stderr)")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            # После загрузки проверяем размер файлов (модель могла попасть в hub/ при скачивании)
            model_cache_path = Path(cache_dir) / 'hub' / model_cache_dir
            if not model_cache_path.exists():
                model_cache_path = Path(cache_dir) / model_cache_dir
            if model_cache_path.exists():
                total_size = sum(f.stat().st_size for f in model_cache_path.rglob('*') if f.is_file())
                total_size_gb = total_size / (1024 ** 3)
                self.logger.info(f"   ✅ Файлы модели загружены, размер: {total_size_gb:.2f} GB")
            
            model_load_time = time.time() - model_start_time
            self.logger.info(f"✅ Файлы модели загружены за {model_load_time:.2f} сек")
            
            # Move model to device
            self.logger.info(f"🚀 Перенос модели на устройство: {self.device}...")
            device_start_time = time.time()
            self.model.to(self.device)
            self.model.eval()
            device_time = time.time() - device_start_time
            self.logger.info(f"✅ Модель перенесена на {self.device} за {device_time:.2f} сек")
            
            total_time = time.time() - tokenizer_start_time
            self.logger.info(f"✅ Модель полностью загружена за {total_time:.2f} сек")
            
            # Log model info
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"📊 Параметров модели: {num_params:,}")
            
            # Log model size
            model_size_gb = self.get_model_size() / (1024 ** 3)
            self.logger.info(f"💾 Размер модели в памяти: {model_size_gb:.2f} GB")
            
            if self.use_hf_generation_config:
                self._apply_generation_config_from_hf()
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            raise
    
    def _apply_generation_config_from_hf(self) -> None:
        """Подставить параметры генерации из model.generation_config (файл generation_config.json в репозитории модели).
        
        Значения из JSON перезаписывают только те поля, где в GenerationConfig не None.
        Поля temperature / top_p / max_new_tokens и т.д. в YAML при use_hf_generation_config не используются.
        """
        gc = getattr(self.model, 'generation_config', None)
        if gc is None:
            self.logger.warning("⚠️  У модели нет generation_config — оставлены запасные значения из кода")
            return
        applied = []
        if getattr(gc, 'temperature', None) is not None:
            self.temperature = float(gc.temperature)
            applied.append(f"temperature={self.temperature}")
        if getattr(gc, 'top_p', None) is not None:
            self.top_p = float(gc.top_p)
            applied.append(f"top_p={self.top_p}")
        tk = getattr(gc, 'top_k', None)
        if tk is not None:
            self.top_k = int(tk)
            applied.append(f"top_k={self.top_k}")
        if getattr(gc, 'repetition_penalty', None) is not None:
            self.repetition_penalty = float(gc.repetition_penalty)
            applied.append(f"repetition_penalty={self.repetition_penalty}")
        if getattr(gc, 'max_new_tokens', None) is not None:
            self.max_new_tokens = int(gc.max_new_tokens)
            applied.append(f"max_new_tokens={self.max_new_tokens}")
        elif getattr(gc, 'max_length', None) is not None:
            # max_length у HF обычно лимит всей последовательности, не число новых токенов — не подставляем в max_new_tokens
            self.logger.debug(
                "generation_config: есть max_length=%s, нет max_new_tokens — остаётся max_new_tokens=%s",
                gc.max_length,
                self.max_new_tokens,
            )
        if getattr(gc, 'do_sample', None) is not None:
            self._do_sample = bool(gc.do_sample)
            applied.append(f"do_sample={self._do_sample}")
        if applied:
            self.logger.info("📋 Параметры генерации из generation_config.json: " + ", ".join(applied))
        else:
            self.logger.warning(
                "⚠️  generation_config.json почти пустой по сэмплингу — проверьте репозиторий модели на Hugging Face"
            )
    
    @staticmethod
    def _token_ids_present(value) -> bool:
        """Проверка, что в generation_config задан pad/eos/bos (int или непустой список ID)."""
        if value is None:
            return False
        if isinstance(value, (list, tuple)):
            return len(value) > 0
        if isinstance(value, int):
            return True
        return False
    
    def _generation_special_token_kwargs(self) -> dict:
        """pad_token_id / eos_token_id / bos_token_id из model.generation_config, иначе из tokenizer.
        
        eos_token_id может быть списком (несколько стоп-токенов) — так и передаём в ``generate``.
        """
        gc = getattr(self.model, 'generation_config', None)
        out = {}
        pad = getattr(gc, 'pad_token_id', None) if gc is not None else None
        out['pad_token_id'] = pad if self._token_ids_present(pad) else self.tokenizer.pad_token_id
        
        eos = getattr(gc, 'eos_token_id', None) if gc is not None else None
        out['eos_token_id'] = eos if self._token_ids_present(eos) else self.tokenizer.eos_token_id
        
        bos = getattr(gc, 'bos_token_id', None) if gc is not None else None
        if self._token_ids_present(bos):
            out['bos_token_id'] = bos
        return out
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Determine optimal dtype: use model's preferred format (bf16/fp16) if GPU supports it.
        
        - CPU: always float32
        - CUDA: prefer model's torch_dtype from config (bf16 for modern models)
        - Fallback to fp16 if model wants bf16 but GPU doesn't support it (e.g. V100)
        """
        if self.device.type != 'cuda':
            return torch.float32
        
        try:
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            model_dtype = getattr(config, 'torch_dtype', None)
            
            if model_dtype is not None:
                # Convert string from config.json to torch dtype (config may store as "bfloat16")
                if isinstance(model_dtype, str):
                    dtype_map = {
                        'bfloat16': torch.bfloat16,
                        'float16': torch.float16,
                        'float32': torch.float32,
                    }
                    model_dtype = dtype_map.get(model_dtype.lower(), torch.float16)
                elif not isinstance(model_dtype, torch.dtype):
                    model_dtype = torch.float16
                
                # Model wants bf16: use it only if GPU supports it (H100, A100, etc.)
                if model_dtype == torch.bfloat16:
                    if torch.cuda.is_bf16_supported():
                        self.logger.info("🎯 Using bf16 (model supports it, GPU supports it)")
                        return torch.bfloat16
                    self.logger.info("⚠️ Model prefers bf16 but GPU doesn't support it, using fp16")
                    return torch.float16
                
                if model_dtype in (torch.float16, torch.float32):
                    self.logger.info(f"🎯 Using {model_dtype} (from model config)")
                    return model_dtype
        except Exception as e:
            self.logger.warning(f"Could not get model dtype from config: {e}")
        
        return torch.float16
    
    def generate(self, prompt: str, context: Optional[List[str]] = None, prompt_template: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        """Generate answer for the given prompt.
        
        Args:
            prompt: Input question/prompt
            context: Optional context (list of strings)
            prompt_template: Optional custom prompt template (defaults to built-in templates)
            
        Returns:
            Generated answer text
        """
        # Check if this is a Qwen3 model (from unsloth) that needs chat template with thinking disabled
        is_qwen3 = 'Qwen3' in self.model_path or 'qwen3' in self.model_path.lower()
        use_chat_template = is_qwen3 and hasattr(self.tokenizer, 'apply_chat_template')
        
        if use_chat_template:
            # Use chat template for Qwen3 models with thinking disabled
            messages = []
            # System message: use system_prompt if provided, else "Context: ..." when context exists
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            elif context and len(context) > 0:
                context_str = "\n".join(context)
                messages.append({"role": "system", "content": f"Context: {context_str}"})
            
            if prompt_template:
                # Use custom prompt template but format it as a user message
                if context and len(context) > 0:
                    context_str = "\n".join(context)
                    user_content = prompt_template.replace("{context}", context_str).replace("{question}", prompt)
                else:
                    user_content = prompt_template.replace("{question}", prompt)
            else:
                user_content = prompt
            
            messages.append({"role": "user", "content": user_content})
            
            # Apply chat template with thinking disabled for Qwen3
            try:
                full_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  # Explicitly disable thinking mode
                )
                self.logger.debug("Using Qwen3 chat template with thinking disabled")
            except Exception as e:
                self.logger.warning(f"Failed to apply chat template: {e}, falling back to simple format")
                use_chat_template = False
        
        if not use_chat_template:
            # Format input using prompt template or default behavior
            if prompt_template:
                # Use custom prompt template from config
                if context and len(context) > 0:
                    context_str = "\n".join(context)
                    full_prompt = prompt_template.replace("{context}", context_str).replace("{question}", prompt)
                else:
                    full_prompt = prompt_template.replace("{question}", prompt)
            else:
                # Fallback to original behavior
                if context and len(context) > 0:
                    context_str = "\n".join(context)
                    full_prompt = f"Context: {context_str}\n\nQuestion: {prompt}\nAnswer:"
                else:
                    full_prompt = f"Question: {prompt}\nAnswer:"
            # Prepend system prompt for non-chat models when provided
            if system_prompt:
                full_prompt = system_prompt + "\n\n" + full_prompt
        
        # Store last prompt for logging
        self.last_prompt = full_prompt
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                if self._do_sample is not None:
                    use_sampling = self._do_sample
                else:
                    gc = getattr(self.model, 'generation_config', None)
                    gc_do = getattr(gc, 'do_sample', None) if gc is not None else None
                    if gc_do is not None:
                        use_sampling = bool(gc_do)
                    else:
                        use_sampling = self.temperature > 0.1
                
                # Создаем словарь параметров для генерации
                generate_kwargs = dict(inputs)  # Копируем inputs
                generate_kwargs.update(self._generation_special_token_kwargs())
                generate_kwargs.update({
                    'max_new_tokens': self.max_new_tokens,
                    'min_new_tokens': 1,  # Минимальная длина ответа
                    'repetition_penalty': self.repetition_penalty,
                    'no_repeat_ngram_size': 2,  # Предотвращает повторение биграмм
                })
                
                if use_sampling:
                    gen_sample = {
                        'temperature': self.temperature,
                        'top_p': self.top_p,
                        'do_sample': True,
                    }
                    if self.top_k is not None and self.top_k > 0:
                        gen_sample['top_k'] = self.top_k
                    generate_kwargs.update(gen_sample)
                else:
                    # Greedy decoding для очень низкой температуры или do_sample=False из generation_config
                    generate_kwargs['do_sample'] = False
                
                # Попытка генерации с обработкой ошибки совместимости autocast
                # Некоторые модели (например, Nanbeige) имеют проблему совместимости с PyTorch 2.2+
                # где torch.is_autocast_enabled() не принимает аргументы
                try:
                    outputs = self.model.generate(**generate_kwargs)
                except (TypeError, AttributeError) as e:
                    error_msg = str(e)
                    if "is_autocast_enabled() takes no arguments" in error_msg or "is_autocast_enabled" in error_msg:
                        # Исправление для моделей с проблемой совместимости autocast
                        # Отключаем autocast явно через контекстный менеджер
                        self.logger.warning(f"⚠️  Обнаружена проблема совместимости autocast, отключаю autocast для генерации")
                        # Используем torch.cuda.amp.autocast с enabled=False для отключения autocast
                        with torch.cuda.amp.autocast(enabled=False):
                            outputs = self.model.generate(**generate_kwargs)
                    else:
                        # Передаем ошибку дальше, если это не проблема autocast
                        raise
            
            # Decode output - extract only newly generated tokens (most reliable)
            input_length = inputs['input_ids'].shape[1]
            output_ids = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            # Fallback: if token-based extraction gives nothing, try text-based parsing
            if not answer:
                if use_chat_template:
                    if "<|im_start|>assistant" in generated_text:
                        answer = generated_text.split("<|im_start|>assistant")[-1].strip()
                        answer = re.sub(r'<\|im_end\|>.*$', '', answer, flags=re.DOTALL).strip()
                    elif "assistant" in generated_text.lower():
                        parts = re.split(r'assistant\s*:?\s*', generated_text, flags=re.IGNORECASE)
                        if len(parts) > 1:
                            answer = parts[-1].strip()
                else:
                    if "Answer:" in generated_text:
                        answer = generated_text.split("Answer:")[-1].strip()
                    else:
                        answer = generated_text[len(full_prompt):].strip()
            
            # Clean up the answer - убираем лишние пробелы и переносы строк
            # НЕ обрезаем по первой строке, так как ответ может быть многострочным
            answer = answer.strip()
            
            # Убираем повторяющиеся пробелы и переносы строк
            answer = re.sub(r'\s+', ' ', answer).strip()
            
            # Remove thinking-related tags (Qwen3 thinking mode output)
            # Remove <think>...</think> tags and their content (including empty tags)
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.IGNORECASE | re.DOTALL)
            # Also handle self-closing tags
            answer = re.sub(r'<think\s*/>', '', answer, flags=re.IGNORECASE)
            # Remove any remaining <think> or </think> tags
            answer = re.sub(r'</?think>', '', answer, flags=re.IGNORECASE)
            
            # Remove thinking-related patterns if they still appear (fallback cleanup)
            thinking_patterns = [
                r'Step-by-Step Explanation:.*?(?=\w)',
                r'Let me think.*?(?=\w)',
                r'First,.*?(?=\w)',
            ]
            for pattern in thinking_patterns:
                answer = re.sub(pattern, '', answer, flags=re.IGNORECASE | re.DOTALL)
            
            # Final cleanup - remove extra spaces that might have been created
            answer = re.sub(r'\s+', ' ', answer).strip()
            
            # Логируем полный сгенерированный текст для отладки (первые несколько раз)
            if not hasattr(self, '_debug_log_count'):
                self._debug_log_count = 0
            if self._debug_log_count < 3:
                self.logger.debug(f"Полный сгенерированный текст: {generated_text}")
                self.logger.debug(f"Извлеченный ответ: {answer}")
                self._debug_log_count += 1
            
            return answer if answer else "No answer generated"
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return "Error generating answer"
    
    def get_model_size(self) -> int:
        """Get model size in bytes.
        
        Returns:
            Model size in bytes
        """
        return sum(p.nelement() * p.element_size() for p in self.model.parameters())
    
    def get_prompt_template(self, with_context: bool = False) -> str:
        """Get prompt template used by the model.
        
        Args:
            with_context: Whether to show template with or without context
            
        Returns:
            Prompt template string
        """
        if with_context:
            return "Context: {context}\n\nQuestion: {question}\nAnswer:"
        else:
            return "Question: {question}\nAnswer:"

