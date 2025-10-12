from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
from statistics import mean
import json
import time
from clearml import Task, Logger
from omegaconf import OmegaConf

from ..models.base import BaseModel
from ..retrievers.base import BaseRetriever
from ..data.base import BaseDataset, DatasetItem
from .metrics import TokenRecallCalculator
from ..utils.memory_tracker import MemoryTracker
from ..utils.predictions_tracker import PredictionsTracker
from ..utils.logger_wrapper import LoggerWrapper
from ..utils.clearml_config import (
    setup_clearml_environment, 
    create_clearml_task, 
    get_clearml_logger,
    log_experiment_config,
    log_predictions_to_clearml,
    log_metrics_to_clearml
)

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
    model: Optional[Any] = None

class ExperimentRunner:
    """Main class for running experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metric_calculator = TokenRecallCalculator()
        self.memory_tracker = MemoryTracker(Path(config.output_dir))
        self.predictions_tracker = PredictionsTracker(Path(config.output_dir))
        
    def setup_experiment(self, use_clearml: bool = True):
        """Initialize ClearML, create directories, etc."""
        if use_clearml:
            # Настраиваем ClearML с использованием .env файла
            setup_clearml_environment()
            
            # Создаем ClearML задачу
            self.task = create_clearml_task(
                project_name="slm-experiments",
                task_name=self.config.name,
                tags=[self.config.model_config.get('name', 'unknown'), 
                      self.config.dataset_config.get('name', 'unknown'),
                      self.config.context_type]
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
            # Режим без ClearML
            self.task = None
            python_logger = logging.getLogger(__name__)
            self.logger = LoggerWrapper(python_logger)
            self.logger.info("🚀 Начало эксперимента (без ClearML)")
            self.logger.info(f"📁 Директория результатов: {self.config.output_dir}")
            self.logger.info(f"🤖 Модель: {self.config.model_config.get('name', 'unknown')}")
            self.logger.info(f"📊 Датасет: {self.config.dataset_config.get('name', 'unknown')}")
            self.logger.info(f"🔍 Режим: {self.config.context_type}")
        
        # Создаем директорию для результатов
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self, model: BaseModel, retriever: BaseRetriever, dataset: BaseDataset, use_clearml: bool = True):
        """Run the experiment."""
        self.setup_experiment(use_clearml=use_clearml)
        
        # Initial memory state
        self.memory_tracker.log_memory("system", "experiment_start")
        
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
        
        # Сохраняем результаты
        self._save_results(metrics)
        
        # Финальное состояние памяти и сохранение лога
        self.memory_tracker.log_memory("system", "experiment_end")
        self.memory_tracker.save_log()
        
        # Сохраняем предсказания и загружаем как артефакт
        self.predictions_tracker.save_predictions()
        
        # Логируем завершение эксперимента
        self.logger.report_text("✅ Эксперимент успешно завершен!")
        self.logger.report_text(f"📈 Токен-реколл: {metrics.get('token_recall', 0):.4f}")
        self.logger.report_text(f"📊 Количество примеров: {metrics.get('num_examples', 0)}")
        
    def _get_context(self, item: DatasetItem, retriever: Optional[BaseRetriever]) -> List[str]:
        """Get context based on experiment mode."""
        if not self.config.use_retriever:
            if self.config.context_type == "none":
                return []
            elif self.config.context_type == "oracle":
                return [item.context] if item.context else []
        else:
            if not retriever:
                raise ValueError("Retriever is required for retriever_context mode")
            contexts = retriever.retrieve(
                item.question, 
                top_k=self.config.top_k
            )
            return [c.context for c in contexts if c.score >= self.config.min_score]

    def _evaluate(self, model: BaseModel, retriever: Optional[BaseRetriever], 
                 dataset: BaseDataset) -> Dict[str, float]:
        """Evaluate model performance using token recall metric."""
        recalls = []
        processed = 0
        
        # Get total number of examples for progress tracking
        eval_data = list(dataset.get_eval_data())
        total_examples = len(eval_data)
        self.logger.info(f"📊 Всего примеров для обработки: {total_examples}")
        
        for item in eval_data:
            if not item.answer:  # Skip items without ground truth
                continue
            
            if self.config.max_samples and processed >= self.config.max_samples:
                break
                
            # Track memory for retrieval
            if retriever:
                self.memory_tracker.log_memory("retriever", "before_retrieve")
            
            # Get context based on experiment mode
            contexts = self._get_context(item, retriever)
            
            if retriever:
                self.memory_tracker.log_memory("retriever", "after_retrieve")
            
            # Track memory for model inference
            self.memory_tracker.log_memory("model", "before_generate")
            
            # Generate answer
            predicted_answer = model.generate(
                prompt=item.question,
                context=contexts
            )
            
            self.memory_tracker.log_memory("model", "after_generate")
            
            # Calculate token recall
            recall = self.metric_calculator.calculate_recall(
                predicted=predicted_answer,
                ground_truth=item.answer
            )
            recalls.append(recall)
            
            # Track prediction
            self.predictions_tracker.add_prediction(
                question_id=item.metadata.get('id', str(processed)),
                question=item.question,
                predicted_answer=predicted_answer,
                ground_truth=item.answer,
                contexts=contexts,
                context_type=self.config.context_type,
                model_name=self.config.model_config.get('name', 'unknown'),
                token_recall=recall,
                metadata={
                    'dataset': self.config.dataset_config.get('name', 'unknown'),
                    **item.metadata
                }
            )
            
            # Log individual example progress (создает график прогресса)
            if hasattr(self.logger, 'report_scalar'):
                self.logger.report_scalar(
                    title="Training Progress",
                    series="token_recall",
                    value=recall,
                    iteration=processed
                )
            
            # Increment processed counter
            processed += 1
            
            # Clear memory periodically and log progress AFTER incrementing
            if processed % 100 == 0:
                self.memory_tracker.clear_memory()
                self.logger.info(f"📊 Обработано примеров: {processed}/{total_examples} ({processed/total_examples*100:.1f}%)")
        
        # Calculate average recall
        avg_recall = mean(recalls) if recalls else 0.0
        
        return {
            "token_recall": avg_recall,
            "num_examples": len(recalls)
        }
    
    def _save_results(self, metrics: Dict[str, float]):
        """Save experiment results to disk and upload to ClearML."""
        results_file = self.config.output_dir / "results.json"
        
        # Сохраняем результаты локально
        with open(results_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        if self.task is not None:
            # Устанавливаем output_uri для артефактов в S3
            import os
            s3_output_uri = f"s3://clearml-artifacts/{self.config.name}"
            
            # Загружаем результаты как артефакт в ClearML
            self.task.upload_artifact(
                name="experiment_results",
                artifact_object=results_file,
                metadata={
                    "experiment_name": self.config.name,
                    "model": self.config.model_config.get('name', 'unknown'),
                    "dataset": self.config.dataset_config.get('name', 'unknown'),
                    "retriever": self.config.retriever_config.get('name', 'unknown'),
                    "timestamp": time.time()
                }
            )
            
            # Загружаем предсказания как артефакт
            predictions_file = self.config.output_dir / "predictions.json"
            if predictions_file.exists():
                self.task.upload_artifact(
                    name="model_predictions",
                    artifact_object=predictions_file,
                    metadata={
                        "experiment_name": self.config.name,
                        "num_predictions": len(self.predictions_tracker.predictions),
                        "timestamp": time.time()
                    }
                )
            
            # Также загружаем memory usage если есть
            memory_file = self.config.output_dir / "memory_usage.json"
            if memory_file.exists():
                self.task.upload_artifact(
                    name="memory_usage",
                    artifact_object=memory_file,
                    metadata={
                        "experiment_name": self.config.name,
                        "timestamp": time.time()
                    }
                )
            
            # Опционально удаляем локальные артефакты для экономии места
            cleanup_local = self.config.model_config.get('cleanup_local_artifacts', False)
            if cleanup_local:
                self.logger.info("🗑️  Очистка локальных артефактов после загрузки в S3...")
                import os
                try:
                    if results_file.exists():
                        os.remove(results_file)
                    if predictions_file.exists():
                        os.remove(predictions_file)
                    if memory_file.exists():
                        os.remove(memory_file)
                    self.logger.info("✅ Локальные артефакты удалены (сохранены в S3)")
                except Exception as e:
                    self.logger.warning(f"⚠️  Не удалось удалить локальные артефакты: {e}")
            
            # Логируем предсказания в ClearML
            if hasattr(self.predictions_tracker, 'predictions') and self.predictions_tracker.predictions:
                log_predictions_to_clearml(self.logger, self.predictions_tracker.predictions)
            
            # Логируем метрики в ClearML
            log_metrics_to_clearml(self.logger, metrics)
        else:
            # Режим без ClearML - только локальное логирование
            self.logger.info("📊 Финальные результаты эксперимента:")
            for metric_name, value in metrics.items():
                self.logger.info(f"  {metric_name}: {value:.4f}")
