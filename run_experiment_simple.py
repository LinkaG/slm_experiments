#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Универсальный скрипт для запуска экспериментов с использованием configs/config.yaml
"""

import logging
import traceback
from pathlib import Path
from omegaconf import OmegaConf
import json
import time
from statistics import mean

# Импортируем компоненты проекта
from src.experiment.runner import ExperimentRunner, ExperimentConfig
from src.models import get_model
from src.retrievers import get_retriever
from src.data import get_dataset
from src.data.base import DatasetItem

def setup_logging():
    """Настройка логирования."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('experiment.log')
        ]
    )
    return logging.getLogger(__name__)

def run_experiment_with_config():
    """Запуск эксперимента с использованием configs/config.yaml"""
    logger = setup_logging()
    logger.info("🚀 Запуск эксперимента с основной конфигурацией")

    try:
        # Загружаем основную конфигурацию с помощью Hydra
        from hydra import initialize, compose
        from hydra.core.global_hydra import GlobalHydra
        
        # Очищаем предыдущие инициализации Hydra
        GlobalHydra.instance().clear()
        
        # Инициализируем Hydra
        with initialize(config_path="configs", version_base=None):
            config = compose(config_name="config")
        
        logger.info("✅ Конфигурация загружена")

        # Проверяем наличие данных
        train_path = Path(config.dataset.train_path)
        eval_path = Path(config.dataset.eval_path)

        if not train_path.exists() or not eval_path.exists():
            logger.error("❌ Файлы данных не найдены. Проверьте пути в конфиге.")
            return False
        logger.info("✅ Файлы данных найдены")

        output_dir = Path(config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Директория результатов: {output_dir}")

        logger.info("📊 Загрузка данных...")
        dataset = get_dataset(config.dataset)
        
        # Ретривер нужен только для некоторых режимов
        retriever = None
        if config.experiment_mode.get('use_retriever', False):
            retriever = get_retriever(config.retriever)

        model = get_model(config.model)

        eval_data = list(dataset.get_eval_data())
        logger.info(f"📈 Количество примеров: {len(eval_data)}")
        logger.info(f"🤖 Модель: {config.model.name}")
        logger.info(f"🔍 Режим: {config.experiment_mode.name}")

        logger.info("▶️ Запуск эксперимента...")
        
        # Создаем ExperimentConfig для ExperimentRunner
        experiment_config = ExperimentConfig(
            name=config.experiment.name,
            model_config=dict(config.model),
            retriever_config=dict(config.retriever) if hasattr(config, 'retriever') else {},
            dataset_config=dict(config.dataset),
            metrics_config={"metrics": config.experiment_mode.metrics} if hasattr(config.experiment_mode, 'metrics') else {},
            output_dir=output_dir,
            model=model,  # Передаем модель
            max_samples=config.experiment_mode.get('max_samples'),
            use_retriever=config.experiment_mode.get('use_retriever', False),
            context_type=config.experiment_mode.get('context_type', 'none')
        )

        runner = ExperimentRunner(experiment_config)
        
        # Переопределяем setup_experiment для отключения ClearML
        def mock_setup_experiment(self):
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"🚀 Начало эксперимента (без ClearML)")
            self.logger.info(f"📁 Директория результатов: {self.config.output_dir}")
            self.logger.info(f"🤖 Модель: {self.config.model_config.get('name', 'unknown')}")
            self.logger.info(f"📊 Датасет: {self.config.dataset_config.get('name', 'unknown')}")
            self.logger.info(f"🔍 Режим: {self.config.context_type}")

        ExperimentRunner.setup_experiment = mock_setup_experiment
        
        # Переопределяем _save_results для сохранения только локально
        def mock_save_results(self, metrics):
            results_file = self.config.output_dir / "results.json"
            with open(results_file, "w", encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            self.logger.info(f"✅ Результаты сохранены в: {results_file}")
            
            # Добавляем дополнительные метрики для вывода
            self.logger.info(f"📊 Token Recall: {metrics.get('token_recall', 0.0):.3f}")
            self.logger.info(f"📈 Количество примеров: {metrics.get('num_examples', 0)}")
            self.logger.info(f"⏱️ Время выполнения: {metrics.get('duration_seconds', 0.0):.2f} секунд")

        ExperimentRunner._save_results = mock_save_results
        
        # Переопределяем logger для отключения ClearML
        def mock_logger_report_scalar(self, title, series, value, iteration=None):
            # Используем стандартный logger вместо self.logger
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"📊 {title}/{series}: {value}")
        
        def mock_logger_report_text(self, text):
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"📝 {text}")
        
        # Мокаем logger
        import types
        runner.logger.report_scalar = types.MethodType(mock_logger_report_scalar, runner.logger)
        runner.logger.report_text = types.MethodType(mock_logger_report_text, runner.logger)

        runner.run(model, retriever, dataset)

        logger.info("🎉 Эксперимент завершен успешно!")
        logger.info(f"📁 Результаты сохранены в {output_dir}/")

    except Exception as e:
        logger.error(f"❌ Ошибка при запуске эксперимента: {e}")
        logger.error(traceback.format_exc())
        logger.info("💥 Эксперимент завершился с ошибкой!")
        return False
    return True

if __name__ == "__main__":
    run_experiment_with_config()
