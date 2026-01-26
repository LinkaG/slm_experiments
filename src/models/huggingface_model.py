"""HuggingFace model implementation."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import logging
import re

from .base import BaseModel


class HuggingFaceModel(BaseModel):
    """HuggingFace transformer model implementation."""
    
    def __init__(self, config):
        """Initialize HuggingFace model.
        
        Args:
            config: Model configuration containing:
                - model_path: HuggingFace model name or path
                - max_length: Maximum sequence length
                - temperature: Sampling temperature
                - top_p: Nucleus sampling parameter
                - device: Device to use (cuda/cpu)
                - batch_size: Batch size for inference
                - use_flash_attention: Whether to use flash attention
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.last_prompt = None  # Store last used prompt for logging
        
        # Extract config parameters
        self.model_path = config.get('model_path', 'gpt2')
        self.max_length = config.get('max_length', 512)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.9)
        self.repetition_penalty = config.get('repetition_penalty', 1.1)  # Предотвращает зацикливание
        self.max_new_tokens = config.get('max_new_tokens', 150)  # Максимальная длина ответа
        self.batch_size = config.get('batch_size', 32)
        
        # Device configuration
        device_config = config.get('device', 'cuda')
        if device_config == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"🎯 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("💻 Using CPU")
        
        # Load model and tokenizer
        self.logger.info(f"📦 Loading model: {self.model_path}")
        
        try:
            # Load tokenizer
            # Используем use_fast=False для совместимости со старыми версиями tokenizers
            # Если возникает ошибка ModelWrapper, это поможет использовать медленный токенайзер
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    use_fast=True
                )
            except Exception as e:
                self.logger.warning(f"Не удалось загрузить быстрый токенайзер: {e}")
                self.logger.info("Пробуем использовать медленный токенайзер...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    use_fast=False
                )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if self.device.type == 'cuda' else torch.float32,
            }
            
            # Add flash attention if requested and available
            if config.get('use_flash_attention', False):
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                self.logger.info("⚡ Using Flash Attention 2")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"✅ Model loaded successfully on {self.device}")
            
            # Log model info
            num_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"📊 Model parameters: {num_params:,}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading model: {e}")
            raise
    
    def generate(self, prompt: str, context: Optional[List[str]] = None, prompt_template: Optional[str] = None) -> str:
        """Generate answer for the given prompt.
        
        Args:
            prompt: Input question/prompt
            context: Optional context (list of strings)
            prompt_template: Optional custom prompt template (defaults to built-in templates)
            
        Returns:
            Generated answer text
        """
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
                # Для низкой температуры используем более детерминированную генерацию
                use_sampling = self.temperature > 0.1
                
                # Создаем словарь параметров для генерации
                generate_kwargs = dict(inputs)  # Копируем inputs
                generate_kwargs.update({
                    'max_new_tokens': self.max_new_tokens,
                    'min_new_tokens': 1,  # Минимальная длина ответа
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'repetition_penalty': self.repetition_penalty,
                    'no_repeat_ngram_size': 2,  # Предотвращает повторение биграмм
                })
                
                if use_sampling:
                    # Сэмплирование для температуры > 0.1
                    generate_kwargs.update({
                        'temperature': self.temperature,
                        'top_p': self.top_p,
                        'do_sample': True,
                    })
                else:
                    # Greedy decoding для очень низкой температуры
                    generate_kwargs['do_sample'] = False
                
                outputs = self.model.generate(**generate_kwargs)
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part (after "Answer:")
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(full_prompt):].strip()
            
            # Clean up the answer - убираем лишние пробелы и переносы строк
            # НЕ обрезаем по первой строке, так как ответ может быть многострочным
            answer = answer.strip()
            
            # Убираем повторяющиеся пробелы и переносы строк
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

