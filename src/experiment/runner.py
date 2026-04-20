from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
from statistics import mean
import json
import time
from omegaconf import OmegaConf
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# clearml_config кладёт clearml.conf.docker в ~/.clearml.conf до import clearml — иначе SDK
# инициализируется без MinIO (HOME=/tmp в Docker с --user). Не импортировать clearml раньше этого.
from ..utils.clearml_config import (
    setup_clearml_environment,
    create_clearml_task,
    get_clearml_logger,
    log_experiment_config,
    log_predictions_to_clearml,
    log_metrics_to_clearml,
)
from clearml import Task, Logger

from ..models.base import BaseModel
from ..retrievers.base import BaseRetriever
from ..data.base import BaseDataset, DatasetItem
from .metrics import TokenRecallCalculator, get_ground_truth_for_recall
from ..utils.memory_tracker import MemoryTracker
from ..utils.predictions_tracker import PredictionsTracker
from ..utils.logger_wrapper import LoggerWrapper
from ..utils.local_logger import LocalLogger

@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    model_config: Dict[str, Any]
    retriever_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    metrics_config: Dict[str, Any]
    output_dir: Path
    max_samples: Optional[int] = None
    use_retriever: bool = False
    context_type: str = "none"
    prompt_template: str = "Question: {question}\nAnswer:"  # default prompt
    system_prompt: Optional[str] = None  # optional system message for chat models
    model: Optional[Any] = None
    clearml_project: str = "no_context"
    # Полный resolved-конфиг Hydra для {name}_conf.json (опционально)
    hydra_config: Optional[Dict[str, Any]] = None

class ExperimentRunner:
    """Main class for running experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Инициализируем metric_calculator позже, когда модель будет загружена
        # чтобы использовать тот же tokenizer, что и модель
        self.metric_calculator = None
        self.memory_tracker = MemoryTracker(Path(config.output_dir))
        self.predictions_tracker = PredictionsTracker(Path(config.output_dir), config.name)

    def _artifact_stem(self) -> str:
        """Имя эксперимента, безопасное для имён файлов."""
        return self.config.name.replace("/", "_").replace("\\", "_")

    def setup_experiment(self, use_clearml: bool = True):
        """Initialize ClearML, create directories, etc."""
        self.use_clearml = use_clearml  # Сохраняем для использования в других методах
        if use_clearml:
            # Настраиваем ClearML с использованием .env файла
            setup_clearml_environment()
            
            # Создаем ClearML задачу
            self.task = create_clearml_task(
                project_name=self.config.clearml_project,
                task_name=self.config.name,
                tags=[self.config.model_config.get('name', 'unknown'), 
                      self.config.dataset_config.get('name', 'unknown'),
                      self.config.context_type]
            )
            
            # Не задаём task.output_uri для S3: ClearML делает check_write_permissions через boto3
            # без надёжного MinIO endpoint из конфига (часто InvalidAccessKeyId к AWS).
            # Файлы в MinIO — через _upload_file_to_minio; в ClearML — upload_artifact (путь/ссылка).
            self.logger.info(
                "📦 ClearML output_uri отключён; загрузка артефактов в MinIO — напрямую (boto3), регистрация в задаче"
            )
            
            # Конвертируем OmegaConf объекты в обычные Python типы для ClearML
            config_dict = {
                "model": self.config.model_config,
                "retriever": self.config.retriever_config,
                "dataset": self.config.dataset_config,
                "metrics": self.config.metrics_config,
                "experiment": {
                    "name": self.config.name,
                    "max_samples": self.config.max_samples,
                    "use_retriever": self.config.use_retriever,
                    "context_type": self.config.context_type
                },
                "prompt": {
                    "template": self.config.prompt_template
                }
            }
            
            # Конвертируем весь словарь целиком
            # Проверяем, нужно ли конвертировать (если элементы уже dict, то пропускаем)
            try:
                config_plain = OmegaConf.to_container(config_dict, resolve=True)
            except (ValueError, TypeError):
                # Если config_dict уже содержит обычные dict, используем его как есть
                config_plain = config_dict
            
            # Логируем конфигурацию эксперимента
            self.task.connect(config_plain)
            
            # Настраиваем логирование
            clearml_logger = get_clearml_logger()
            self.logger = LoggerWrapper(clearml_logger)
            
            # Логируем полную конфигурацию
            full_config = {
                "model": self.config.model_config,
                "retriever": self.config.retriever_config,
                "dataset": self.config.dataset_config,
                "metrics": self.config.metrics_config,
                "experiment": {
                    "name": self.config.name,
                    "output_dir": str(self.config.output_dir),
                    "max_samples": self.config.max_samples,
                    "use_retriever": self.config.use_retriever,
                    "context_type": self.config.context_type
                },
                "prompt": {
                    "template": self.config.prompt_template
                }
            }
            # Конвертируем в обычные Python типы
            try:
                full_config_plain = OmegaConf.to_container(full_config, resolve=True)
            except (ValueError, TypeError):
                # Если full_config уже содержит обычные dict, используем его как есть
                full_config_plain = full_config
            log_experiment_config(self.logger, full_config_plain)
        else:
            # Режим без ClearML - используем LocalLogger
            self.task = None
            self.local_logger = LocalLogger(self.config.output_dir)
            python_logger = logging.getLogger(__name__)
            # Объединяем LocalLogger и стандартный logger через wrapper
            self.logger = LoggerWrapper(self.local_logger)
            self.logger.info("🚀 Начало эксперимента (без ClearML)")
            self.logger.info(f"📁 Директория результатов: {self.config.output_dir}")
            self.logger.info(f"🤖 Модель: {self.config.model_config.get('name', 'unknown')}")
            self.logger.info(f"📊 Датасет: {self.config.dataset_config.get('name', 'unknown')}")
            self.logger.info(f"🔍 Режим: {self.config.context_type}")
            
            # Сохраняем конфигурацию в локальный логгер
            full_config = {
                "model": self.config.model_config,
                "retriever": self.config.retriever_config,
                "dataset": self.config.dataset_config,
                "metrics": self.config.metrics_config,
                "experiment": {
                    "name": self.config.name,
                    "output_dir": str(self.config.output_dir),
                    "max_samples": self.config.max_samples,
                    "use_retriever": self.config.use_retriever,
                    "context_type": self.config.context_type
                },
                "prompt": {
                    "template": self.config.prompt_template
                }
            }
            try:
                full_config_plain = OmegaConf.to_container(full_config, resolve=True)
            except (ValueError, TypeError):
                full_config_plain = full_config
            self.local_logger.config = full_config_plain
        
        # Создаем директорию для результатов
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self, model: BaseModel, retriever: BaseRetriever, dataset: BaseDataset, use_clearml: bool = True):
        """Run the experiment."""
        self.setup_experiment(use_clearml=use_clearml)
        
        # Инициализируем metric_calculator с токенизатором модели
        # Это важно для правильного подсчета recall - нужно использовать тот же токенизатор
        if hasattr(model, 'tokenizer') and model.tokenizer is not None:
            self.metric_calculator = TokenRecallCalculator(tokenizer=model.tokenizer)
            self.logger.info("✅ TokenRecallCalculator инициализирован с токенизатором модели")
        else:
            # Fallback: используем токенизатор из конфига модели
            model_path = self.config.model_config.get('model_path', 'bert-base-uncased')
            self.metric_calculator = TokenRecallCalculator(tokenizer_name=model_path)
            self.logger.warning(f"⚠️  Используется токенизатор из конфига: {model_path}")
        
        # Initial memory state
        self.memory_tracker.log_memory("system", "experiment_start")
        
        # Log prompt template from config
        if hasattr(self.logger, 'report_text'):
            self.logger.report_text("📝 Шаблон промпта:")
            self.logger.report_text("")
            self.logger.report_text("```")
            self.logger.report_text(self.config.prompt_template)
            self.logger.report_text("```")
        else:
            self.logger.info(f"Prompt template: {self.config.prompt_template}")
        
        # Log basic info as single values (не создают графики)
        if hasattr(self.logger, 'report_single_value'):
            self.logger.report_single_value("model_size_bytes", model.get_model_size())
            if retriever is not None:
                self.logger.report_single_value("retriever_index_size", retriever.get_index_size())
            # Log dataset stats
            for key, value in dataset.get_dataset_stats().items():
                self.logger.report_single_value(f"dataset_{key}", value)
        else:
            # Fallback для режима без ClearML
            self.logger.info(f"Model size: {model.get_model_size()}")
            if retriever is not None:
                self.logger.info(f"Retriever index size: {retriever.get_index_size()}")
            for key, value in dataset.get_dataset_stats().items():
                self.logger.info(f"Dataset {key}: {value}")
        
        # Запускаем оценку
        self.logger.report_text("🔄 Начинаем оценку модели...")
        start_time = time.time()
        
        metrics = self._evaluate(model, retriever, dataset)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Логируем время выполнения
        if hasattr(self.logger, 'report_single_value'):
            self.logger.report_single_value("duration_seconds", duration)
        self.logger.report_text(f"⏱️ Время выполнения: {duration:.2f} секунд")
        
        # Добавляем информацию о памяти и размерах в метрики
        metrics['duration_seconds'] = duration
        metrics['model_size_bytes'] = model.get_model_size()
        metrics['model_size_mb'] = model.get_model_size() / (1024 * 1024)
        
        if retriever is not None:
            metrics['retriever_index_size_bytes'] = retriever.get_index_size()
            metrics['retriever_index_size_mb'] = retriever.get_index_size() / (1024 * 1024)
        else:
            metrics['retriever_index_size_bytes'] = 0
            metrics['retriever_index_size_mb'] = 0

        # Финальное состояние памяти и сохранение лога
        self.memory_tracker.log_memory("system", "experiment_end")
        self.memory_tracker.save_log()

        # Пиковые замеры + детальный лог памяти в один объект метрик
        if self.memory_tracker.peak_stats:
            peak_memory = self.memory_tracker.peak_stats.to_dict()
            metrics.update(peak_memory)

            self.logger.report_text("💾 Пиковое использование памяти:")
            self.logger.report_text(f"  CPU RAM: {peak_memory['cpu_ram_used_mb']:.2f} MB")
            if peak_memory['gpu_ram_peak_mb'] > 0:
                self.logger.report_text(f"  GPU RAM (peak): {peak_memory['gpu_ram_peak_mb']:.2f} MB")
                self.logger.report_text(f"  GPU RAM (reserved): {peak_memory['reserved_gpu_ram_mb']:.2f} MB")
            if peak_memory.get('gpu_power_draw_w', 0) > 0:
                power_limit = peak_memory.get('gpu_power_limit_w', 0)
                power_pct = (peak_memory['gpu_power_draw_w'] / power_limit * 100) if power_limit > 0 else 0
                self.logger.report_text(
                    f"  GPU Power (peak): {peak_memory['gpu_power_draw_w']:.2f}W / "
                    f"{power_limit:.2f}W ({power_pct:.1f}%)"
                )

        # Детальный таймлайн памяти — только в memory_usage.json (не дублируем в metrics.json)

        # Конфиг + метрики на диск и в S3
        conf_path = self._save_conf_file()
        metrics_path = self._save_metrics_file(metrics)
        self._upload_artifacts_bundle(metrics_path, conf_path)
        
        # Сохраняем предсказания и загружаем как артефакт в MinIO
        # Передаем callback для прямой загрузки в MinIO через Docker сеть
        predictions_path = self.predictions_tracker.predictions_json_path
        if self.use_clearml and self.task is not None:
            predictions_s3_path = self.predictions_tracker.save_predictions(
                upload_to_minio_callback=self._upload_to_minio_direct
            )
            if predictions_s3_path:
                self._register_clearml_artifact(
                    name="model_predictions",
                    s3_path=predictions_s3_path,
                    local_file=predictions_path,
                    metadata={
                        "experiment_name": self.config.name,
                        "num_predictions": len(self.predictions_tracker.predictions),
                        "timestamp": time.time(),
                        "storage": "MinIO S3",
                        "s3_path": predictions_s3_path,
                    },
                )
        else:
            self.predictions_tracker.save_predictions()

        # Логируем завершение эксперимента
        self.logger.report_text("✅ Эксперимент успешно завершен!")
        self.logger.report_text(f"📈 Substring recall: {metrics.get('substring_recall', 0):.4f}")
        self.logger.report_text(f"📈 Token recall: {metrics.get('token_recall', 0):.4f}")
        self.logger.report_text(f"📊 Количество примеров: {metrics.get('num_examples', 0)}")
        self.logger.report_text(f"📦 Размер модели: {metrics.get('model_size_mb', 0):.2f} MB")
        self.logger.report_text(f"🔍 Размер индекса RAG: {metrics.get('retriever_index_size_mb', 0):.2f} MB")

        self._finalize_logging_and_cleanup(metrics, predictions_path, metrics_path, conf_path)
        
    def _evaluate(self, model: BaseModel, retriever: Optional[BaseRetriever], 
                 dataset: BaseDataset) -> Dict[str, float]:
        """Evaluate model performance using token recall metric."""
        recalls = []
        token_recalls = []
        processed = 0
        logged_prompt_examples = 0  # Track how many prompt examples we've logged
        max_prompt_examples = 20  # Log first 20 prompt examples
        
        # Get total number of examples for progress tracking
        eval_data = list(dataset.get_eval_data())
        total_examples = len(eval_data)
        
        # Подсчитываем количество элементов с ответами (для прогресс-бара)
        valid_examples = sum(1 for item in eval_data if item.answer)
        
        # Определяем максимальное количество примеров для обработки
        max_to_process = valid_examples
        if self.config.max_samples:
            max_to_process = min(valid_examples, self.config.max_samples)
        
        self.logger.info(f"📊 Всего примеров для обработки: {max_to_process} (из {total_examples} в датасете)")
        
        # Создаем прогресс-бар
        if HAS_TQDM:
            pbar = tqdm(
                total=max_to_process,
                desc="Обработка примеров",
                unit="пример",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            pbar = None
        
        for item in eval_data:
            if not item.answer:  # Skip items without ground truth
                continue
            
            if self.config.max_samples and processed >= self.config.max_samples:
                break

            # Режим no_context: без внешнего контекста
            contexts: List[str] = []

            ground_truth_for_recall = get_ground_truth_for_recall(item.metadata, item.answer)

            self.memory_tracker.log_memory("model", "before_generate")
            generate_kwargs = dict(
                prompt=item.question,
                context=contexts,
                prompt_template=self.config.prompt_template
            )
            if self.config.system_prompt is not None:
                generate_kwargs['system_prompt'] = self.config.system_prompt
            predicted_answer = model.generate(**generate_kwargs)
            self.memory_tracker.log_memory("model", "after_generate")
            recall = self.metric_calculator.calculate_recall(
                predicted=predicted_answer, ground_truth=ground_truth_for_recall
            )
            prediction_contexts = contexts
            prediction_prompt = model.last_prompt if hasattr(model, 'last_prompt') else None
            prediction_metadata_extra: Dict[str, Any] = {}

            # Log prompt examples for first few items
            if logged_prompt_examples < max_prompt_examples and prediction_prompt:
                if hasattr(self.logger, 'report_text'):
                    self.logger.report_text(f"\n💬 Пример промпта #{logged_prompt_examples + 1}:")
                    self.logger.report_text("```")
                    self.logger.report_text(prediction_prompt)
                    self.logger.report_text("```")
                    self.logger.report_text(f"Сгенерированный ответ: {predicted_answer}")
                    self.logger.report_text(f"Правильный ответ: {item.answer}")
                logged_prompt_examples += 1
            
            # Recall по подстрокам (текущий) и по токенам (оригинальный)
            recall_substring = recall
            recall_token = self.metric_calculator.calculate_token_recall(
                predicted_answer, ground_truth_for_recall
            )
            recalls.append(recall_substring)
            token_recalls.append(recall_token)
            
            # Track prediction
            self.predictions_tracker.add_prediction(
                question_id=item.metadata.get('id', str(processed)),
                question=item.question,
                predicted_answer=predicted_answer,
                ground_truth=item.answer,
                contexts=prediction_contexts,
                context_type=self.config.context_type,
                model_name=self.config.model_config.get('name', 'unknown'),
                token_recall=recall_substring,
                metadata={
                    'dataset': self.config.dataset_config.get('name', 'unknown'),
                    'token_recall_tokens': recall_token,
                    **item.metadata,
                    **prediction_metadata_extra
                },
                prompt=prediction_prompt
            )
            
            # Log individual example progress (создает график прогресса)
            if hasattr(self.logger, 'report_scalar'):
                self.logger.report_scalar(
                    title="Training Progress",
                    series="substring_recall",
                    value=recall_substring,
                    iteration=processed
                )
                self.logger.report_scalar(
                    title="Training Progress",
                    series="token_recall",
                    value=recall_token,
                    iteration=processed
                )
            
            # Increment processed counter
            processed += 1
            
            # Обновляем прогресс-бар
            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    'recall': f'{mean(recalls)*100:.2f}%' if recalls else '0%',
                    'processed': processed
                })
            
            # Логируем прогресс в текстовый лог периодически
            log_interval = 10  # Логируем каждые 10 примеров
            if processed % log_interval == 0 or processed == 1:
                avg_recall = mean(recalls) if recalls else 0.0
                progress_pct = (processed / max_to_process * 100) if max_to_process > 0 else 0
                self.logger.info(
                    f"📊 Прогресс: {processed}/{max_to_process} ({progress_pct:.1f}%) | "
                    f"Средний recall: {avg_recall*100:.2f}% | "
                    f"Обработано: {processed}"
                )
            
            # Clear memory periodically and log progress AFTER incrementing
            if processed % 100 == 0:
                self.memory_tracker.clear_memory()
                avg_recall = mean(recalls) if recalls else 0.0
                # Дополнительное логирование каждые 100 примеров с информацией о памяти и мощности
                memory_stats = self.memory_tracker.get_current_memory_usage()
                gpu_info = f"GPU RAM: {memory_stats.gpu_ram_used:.1f} MB" if memory_stats.gpu_ram_used else "GPU RAM: N/A"
                power_info = ""
                if memory_stats.gpu_power_draw is not None:
                    power_pct = (memory_stats.gpu_power_draw / memory_stats.gpu_power_limit * 100) if memory_stats.gpu_power_limit else 0
                    power_info = f" | GPU Power: {memory_stats.gpu_power_draw:.1f}W/{memory_stats.gpu_power_limit:.1f}W ({power_pct:.1f}%)"
                self.logger.info(
                    f"💾 Память: CPU RAM: {memory_stats.cpu_ram_used:.1f} MB | {gpu_info}{power_info}"
                )
        
        # Закрываем прогресс-бар
        if pbar:
            pbar.close()
        
        # Calculate average recall
        avg_substring_recall = mean(recalls) if recalls else 0.0
        avg_token_recall = mean(token_recalls) if token_recalls else 0.0
        
        return {
            "substring_recall": avg_substring_recall,
            "token_recall": avg_token_recall,
            "num_examples": len(recalls)
        }
    
    def _save_conf_file(self) -> Path:
        """{experiment}_conf.json — полная конфигурация (Hydra или собранная из ExperimentConfig)."""
        stem = self._artifact_stem()
        path = self.config.output_dir / f"{stem}_conf.json"
        if self.config.hydra_config is not None:
            payload = self.config.hydra_config
        else:
            payload = {
                "model": self.config.model_config,
                "retriever": self.config.retriever_config,
                "dataset": self.config.dataset_config,
                "metrics": self.config.metrics_config,
                "experiment": {
                    "name": self.config.name,
                    "output_dir": str(self.config.output_dir),
                    "max_samples": self.config.max_samples,
                    "use_retriever": self.config.use_retriever,
                    "context_type": self.config.context_type,
                    "clearml_project": self.config.clearml_project,
                },
                "prompt": {
                    "template": self.config.prompt_template,
                    "system_prompt": self.config.system_prompt,
                },
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        self.logger.info(f"📝 Конфигурация сохранена: {path}")
        return path

    def _save_metrics_file(self, metrics: Dict[str, Any]) -> Path:
        """{experiment}_metrics.json — метрики и пиковые скаляры памяти; таймлайн — в memory_usage.json."""
        stem = self._artifact_stem()
        path = self.config.output_dir / f"{stem}_metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
        self.logger.info(f"📊 Метрики сохранены: {path}")
        return path

    def _upload_artifacts_bundle(self, metrics_path: Path, conf_path: Path) -> None:
        """Загрузка metrics и conf в MinIO (при активной ClearML-задаче)."""
        if self.task is None:
            return
        stem = self._artifact_stem()
        self.logger.info("📤 Загрузка metrics и conf в MinIO...")
        meta_base = {
            "experiment_name": self.config.name,
            "model": self.config.model_config.get("name", "unknown"),
            "dataset": self.config.dataset_config.get("name", "unknown"),
            "timestamp": time.time(),
            "storage": "MinIO S3",
        }
        m_s3 = self._upload_to_minio_direct(
            metrics_path, f"experiment_results/{stem}_metrics.json"
        )
        if m_s3:
            self._register_clearml_artifact(
                name="experiment_metrics",
                s3_path=m_s3,
                local_file=metrics_path,
                metadata={**meta_base, "s3_path": m_s3},
            )
        c_s3 = self._upload_to_minio_direct(
            conf_path, f"experiment_results/{stem}_conf.json"
        )
        if c_s3:
            self._register_clearml_artifact(
                name="experiment_config",
                s3_path=c_s3,
                local_file=conf_path,
                metadata={**meta_base, "s3_path": c_s3},
            )

    def _finalize_logging_and_cleanup(
        self,
        metrics: Dict[str, Any],
        predictions_path: Path,
        metrics_path: Path,
        conf_path: Path,
    ) -> None:
        """Текстовые логи ClearML / локальный логгер; опционально удалить три файла после S3."""
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

        if self.task is not None:
            if self.predictions_tracker.predictions:
                log_predictions_to_clearml(self.logger, self.predictions_tracker.predictions)
            log_metrics_to_clearml(self.logger, scalar_metrics)
        elif hasattr(self, "local_logger"):
            self.local_logger.save_all(self.config.name)
            for metric_name, value in scalar_metrics.items():
                self.local_logger.report_single_value(metric_name, value)
            self.logger.info("📊 Финальные результаты эксперимента (скаляры):")
            for metric_name, value in scalar_metrics.items():
                self.logger.info(f"  {metric_name}: {value:.4f}")
            if metrics_path.exists():
                self.local_logger.save_artifact("experiment_metrics", metrics_path)
            if conf_path.exists():
                self.local_logger.save_artifact("experiment_config", conf_path)
            if predictions_path.exists():
                self.local_logger.save_artifact("model_predictions", predictions_path)

        cleanup_local = self.config.model_config.get("cleanup_local_artifacts", False)
        if cleanup_local and self.task is not None:
            import os
            self.logger.info("🗑️  Очистка локальных артефактов после загрузки в S3...")
            for p in (metrics_path, conf_path, predictions_path):
                try:
                    if p.exists():
                        os.remove(p)
                except OSError as e:
                    self.logger.warning(f"⚠️  Не удалось удалить {p}: {e}")
            self.logger.info("✅ Локальные копии удалены (данные в S3)")

    def _upload_to_minio_direct(self, file_path: Path, s3_key: str) -> Optional[str]:
        """Загружает файл напрямую в MinIO через boto3.
        
        Returns:
            S3 path в формате s3://bucket/key или None в случае ошибки
        """
        try:
            import boto3
            from dotenv import load_dotenv
            import os
            
            # Загружаем настройки из .env (если файл доступен)
            # В Docker контейнере .env может быть не доступен, используем значения по умолчанию
            load_dotenv()
            
            # Определяем endpoint: в Docker сети используем имя контейнера, иначе localhost
            # Приоритет: переменная окружения > автоопределение Docker сети > localhost
            endpoint = os.getenv('CLEARML_S3_ENDPOINT')
            if endpoint:
                self.logger.info(f"🔍 Используется endpoint из переменной окружения: {endpoint}")
            else:
                # Пробуем определить автоматически: если доступен minio:9000, используем его
                try:
                    import socket
                    # Проверяем доступность minio через Docker сеть
                    socket.gethostbyname('minio')
                    endpoint = 'http://minio:9000'
                    self.logger.info(f"🔍 Автоопределение: используется Docker сеть endpoint: {endpoint}")
                except (socket.gaierror, OSError):
                    # Если minio недоступен, используем localhost
                    endpoint = 'http://localhost:9000'
                    self.logger.info(f"🔍 Автоопределение: используется localhost endpoint: {endpoint}")
                    self.logger.warning("⚠️  MinIO через Docker сеть недоступен, используется localhost. Убедитесь, что контейнер запущен в сети clearml_backend")
            
            # Создаем S3 клиент для MinIO
            access_key = os.getenv('CLEARML_S3_ACCESS_KEY', 'minioadmin')
            secret_key = os.getenv('CLEARML_S3_SECRET_KEY', 'minioadmin')
            bucket = os.getenv('CLEARML_S3_BUCKET', 'clearml-artifacts')
            
            self.logger.info(f"🔧 Подключение к MinIO: endpoint={endpoint}, bucket={bucket}")
            
            s3_client = boto3.client(
                's3',
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=os.getenv('CLEARML_S3_REGION', 'us-east-1')
            )
            
            # Проверяем доступность MinIO и существование bucket
            try:
                # Пробуем получить список bucket'ов для проверки подключения
                s3_client.list_buckets()
                self.logger.info("✅ Подключение к MinIO успешно")
            except Exception as conn_error:
                self.logger.error(f"❌ Не удалось подключиться к MinIO: {conn_error}")
                self.logger.error(f"   Проверьте, что MinIO доступен по адресу {endpoint}")
                self.logger.error(f"   Убедитесь, что контейнер запущен в Docker сети clearml_backend")
                return None
            
            # Проверяем существование bucket, создаем если нужно
            try:
                s3_client.head_bucket(Bucket=bucket)
                self.logger.info(f"✅ Bucket '{bucket}' существует")
            except Exception as e:
                # Проверяем код ошибки для определения типа проблемы
                error_code = None
                if hasattr(e, 'response') and isinstance(e.response, dict):
                    error_code = e.response.get('Error', {}).get('Code', '')
                elif hasattr(e, 'error_code'):
                    error_code = e.error_code
                
                # 404 означает, что bucket не существует
                if error_code == '404' or '404' in str(e) or 'Not Found' in str(e):
                    # Bucket не существует, создаем его
                    self.logger.info(f"📦 Bucket '{bucket}' не найден, создаем...")
                    try:
                        s3_client.create_bucket(Bucket=bucket)
                        self.logger.info(f"✅ Bucket '{bucket}' успешно создан")
                    except Exception as create_error:
                        self.logger.error(f"❌ Не удалось создать bucket '{bucket}': {create_error}")
                        return None
                else:
                    # Другая ошибка (например, проблемы с доступом)
                    self.logger.warning(f"⚠️  Не удалось проверить bucket '{bucket}': {e}")
                    self.logger.info(f"💡 Пробуем продолжить - bucket может быть создан автоматически при загрузке")
            
            full_s3_key = f"{self.config.name}/{s3_key}"
            
            # Проверяем, что файл существует
            if not file_path.exists():
                self.logger.warning(f"⚠️ Файл не найден: {file_path}")
                return None
            
            file_size = file_path.stat().st_size
            self.logger.info(f"📤 Загрузка файла {file_path} ({file_size} bytes) -> s3://{bucket}/{full_s3_key}")
            
            # Загружаем файл
            s3_client.upload_file(
                str(file_path),
                bucket,
                full_s3_key
            )
            
            s3_path = f"s3://{bucket}/{full_s3_key}"
            self.logger.info(f"✅ Файл успешно загружен в MinIO: {s3_path}")
            return s3_path
            
        except Exception as e:
            self.logger.error(f"❌ Не удалось загрузить файл в MinIO напрямую: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            # Не прерываем выполнение эксперимента из-за ошибки загрузки в MinIO
            self.logger.warning("⚠️  Продолжаем выполнение эксперимента без загрузки в MinIO")
            return None
    
    def _register_clearml_artifact(self, name: str, s3_path: str, local_file: Path, metadata: Dict[str, Any]):
        """Регистрирует артефакт в ClearML с прямым указанием S3 пути в MinIO.
        
        Пробует использовать S3 URI напрямую. Если не работает, создает файл-ссылку
        с информацией о местоположении в MinIO и добавляет прямую ссылку в метаданные.
        """
        try:
            # Добавляем информацию о местоположении файла в MinIO в метаданные
            metadata_with_s3 = metadata.copy()
            metadata_with_s3["s3_path"] = s3_path
            metadata_with_s3["storage_location"] = "MinIO S3"
            metadata_with_s3["storage_url"] = s3_path  # Прямая ссылка на MinIO
            metadata_with_s3["note"] = f"Файл хранится в MinIO по пути: {s3_path}"
            
            # Проверяем output_uri перед загрузкой
            current_output_uri = getattr(self.task, 'output_uri', None)
            if current_output_uri:
                self.logger.info(f"📦 Используется output_uri: {current_output_uri}")
            else:
                self.logger.warning("⚠️  output_uri не установлен! Артефакт может быть загружен в fileserver")
            
            # Пробуем использовать S3 путь напрямую
            # ClearML может понимать S3 URI если output_uri настроен правильно
            try:
                self.task.upload_artifact(
                    name=name,
                    artifact_object=s3_path,  # Пробуем использовать S3 путь напрямую
                    metadata=metadata_with_s3
                )
                self.logger.info(f"✅ Артефакт '{name}' зарегистрирован в ClearML с S3 путем: {s3_path}")
                return
            except Exception as s3_error:
                # Если прямой S3 путь не работает, создаем файл-ссылку
                self.logger.debug(f"Прямой S3 путь не сработал: {s3_error}")
                self.logger.info(f"💡 Создаем файл-ссылку с информацией о MinIO")
            
            # Создаем файл-ссылку с информацией о местоположении в MinIO
            import json as json_lib
            link_file = local_file.parent / f"{name}_minio_link.json"
            
            # Извлекаем bucket и key из s3_path
            # Формат: s3://bucket/key
            s3_parts = s3_path.replace("s3://", "").split("/", 1)
            bucket = s3_parts[0] if len(s3_parts) > 0 else "clearml-artifacts"
            key = s3_parts[1] if len(s3_parts) > 1 else ""
            
            # Получаем MinIO endpoint из переменных окружения
            import os
            minio_endpoint = os.getenv('CLEARML_S3_ENDPOINT', 'http://minio:9000')
            # Убираем http:// или https:// для создания правильной ссылки
            minio_host = minio_endpoint.replace("http://", "").replace("https://", "")
            
            # Создаем прямую ссылку на MinIO (для использования через boto3 или curl)
            minio_direct_url = f"{minio_endpoint}/{bucket}/{key}"
            
            link_info = {
                "artifact_name": name,
                "storage": "MinIO S3",
                "s3_path": s3_path,
                "bucket": bucket,
                "key": key,
                "minio_endpoint": minio_endpoint,
                "minio_host": minio_host,
                "direct_access_url": minio_direct_url,
                "access_method": "Use boto3 or S3 client with endpoint_url",
                "endpoint_url": minio_endpoint,
                "bucket_name": bucket,
                "object_key": key,
                "credentials": {
                    "access_key": os.getenv('CLEARML_S3_ACCESS_KEY', 'minioadmin'),
                    "secret_key": os.getenv('CLEARML_S3_SECRET_KEY', 'minioadmin')
                },
                "note": "Этот файл содержит информацию о местоположении артефакта в MinIO. Используйте s3_path для доступа через boto3 или S3-совместимый клиент.",
                "metadata": metadata_with_s3
            }
            
            with open(link_file, 'w', encoding='utf-8') as f:
                json_lib.dump(link_info, f, indent=2, ensure_ascii=False)
            
            # Загружаем файл-ссылку как артефакт
            # В метаданных указываем прямую ссылку на MinIO
            metadata_with_s3["minio_direct_url"] = minio_direct_url
            metadata_with_s3["minio_access_info"] = {
                "endpoint": minio_endpoint,
                "bucket": bucket,
                "key": key,
                "s3_path": s3_path
            }
            
            self.task.upload_artifact(
                name=name,
                artifact_object=str(link_file),
                metadata=metadata_with_s3
            )
            
            self.logger.info(f"✅ Артефакт '{name}' зарегистрирован в ClearML")
            self.logger.info(f"📦 Файл в MinIO: {s3_path}")
            self.logger.info(f"🔗 Прямая ссылка на MinIO: {minio_direct_url}")
            self.logger.info(f"💡 Используйте метаданные артефакта для доступа к файлу в MinIO")
            
        except Exception as e:
            self.logger.error(f"❌ Не удалось зарегистрировать артефакт в ClearML: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            # Не критично, файл уже в MinIO
