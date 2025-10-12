#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Универсальный скрипт для запуска экспериментов с использованием configs/config.yaml
Поддерживает логирование в ClearML
"""

import logging
import traceback
import argparse
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


def parse_arguments():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(description='Запуск экспериментов с малыми языковыми моделями')
    parser.add_argument('--use-clearml', action='store_true', default=True,
                        help='Использовать ClearML для логирования (по умолчанию: True)')
    parser.add_argument('--no-clearml', action='store_true',
                        help='Отключить ClearML логирование')
    parser.add_argument('--env-file', type=str, default='.env',
                        help='Путь к файлу .env с настройками ClearML')
    
    # Добавляем поддержку Hydra аргументов
    parser.add_argument('--config-path', type=str, default='configs',
                        help='Путь к конфигурационным файлам')
    parser.add_argument('--config-name', type=str, default='config',
                        help='Имя конфигурационного файла')
    
    return parser.parse_args()

def run_experiment_with_config(use_clearml=True, env_file='.env'):
    """Запуск эксперимента с использованием configs/config.yaml"""
    logger = setup_logging()
    
    if use_clearml:
        logger.info("🚀 Запуск эксперимента с ClearML логированием")
    else:
        logger.info("🚀 Запуск эксперимента без ClearML логирования")

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
        
        # Запускаем эксперимент с указанными параметрами
        runner.run(model, retriever, dataset, use_clearml=use_clearml)

        logger.info("🎉 Эксперимент завершен успешно!")
        logger.info(f"📁 Результаты сохранены в {output_dir}/")

    except Exception as e:
        logger.error(f"❌ Ошибка при запуске эксперимента: {e}")
        logger.error(traceback.format_exc())
        logger.info("💥 Эксперимент завершился с ошибкой!")
        return False
    return True

if __name__ == "__main__":
    # Парсим аргументы командной строки
    args = parse_arguments()
    
    # Определяем, использовать ли ClearML
    use_clearml = args.use_clearml and not args.no_clearml
    
    # Запускаем эксперимент
    run_experiment_with_config(use_clearml=use_clearml, env_file=args.env_file)
