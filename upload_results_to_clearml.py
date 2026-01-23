#!/usr/bin/env python3
"""
Скрипт для загрузки локально сохраненных результатов экспериментов в ClearML.
Используется для загрузки результатов, полученных в режиме без ClearML.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

from dotenv import load_dotenv
from clearml import Task, Logger

from src.utils.clearml_config import setup_clearml_environment, create_clearml_task

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_local_results(results_dir: Path) -> Dict[str, Any]:
    """
    Загружает локально сохраненные результаты эксперимента.
    
    Args:
        results_dir: Директория с результатами эксперимента
        
    Returns:
        Словарь с загруженными данными
    """
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Директория результатов не найдена: {results_dir}")
    
    data = {}
    
    # Загружаем метаданные
    metadata_file = results_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            data["metadata"] = json.load(f)
    
    # Загружаем конфигурацию
    config_file = results_dir / "config" / "experiment_config.json"
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            data["config"] = json.load(f)
    
    # Загружаем текстовые логи
    logs_file = results_dir / "logs" / "text_logs.json"
    if logs_file.exists():
        with open(logs_file, "r", encoding="utf-8") as f:
            data["text_logs"] = json.load(f)
    
    # Загружаем скалярные метрики
    scalar_metrics_file = results_dir / "metrics" / "scalar_metrics.json"
    if scalar_metrics_file.exists():
        with open(scalar_metrics_file, "r", encoding="utf-8") as f:
            data["scalar_metrics"] = json.load(f)
    
    # Загружаем одиночные значения
    single_values_file = results_dir / "metrics" / "single_values.json"
    if single_values_file.exists():
        with open(single_values_file, "r", encoding="utf-8") as f:
            data["single_values"] = json.load(f)
    
    # Загружаем таблицы
    tables_file = results_dir / "metrics" / "tables.json"
    if tables_file.exists():
        with open(tables_file, "r", encoding="utf-8") as f:
            data["tables"] = json.load(f)
    
    # Загружаем метаданные артефактов
    artifacts_meta_file = results_dir / "artifacts" / "artifacts_metadata.json"
    if artifacts_meta_file.exists():
        with open(artifacts_meta_file, "r", encoding="utf-8") as f:
            data["artifacts"] = json.load(f)
    
    return data


def upload_to_clearml(results_dir: Path, project_name: str = "slm-experiments", 
                     task_name: Optional[str] = None, tags: Optional[list] = None):
    """
    Загружает локальные результаты в ClearML.
    
    Args:
        results_dir: Директория с результатами эксперимента
        project_name: Название проекта в ClearML
        task_name: Название задачи (если None, берется из metadata)
        tags: Список тегов для задачи
    """
    logger.info(f"📂 Загружаем результаты из {results_dir}")
    
    # Загружаем локальные данные
    data = load_local_results(results_dir)
    
    # Определяем название задачи
    if task_name is None:
        task_name = data.get("metadata", {}).get("experiment_name", "offline_experiment")
    
    # Определяем теги из конфигурации
    if tags is None:
        tags = []
        config = data.get("config", {})
        if "model" in config:
            tags.append(config["model"].get("name", "unknown"))
        if "dataset" in config:
            tags.append(config["dataset"].get("name", "unknown"))
        if "experiment" in config:
            tags.append(config["experiment"].get("context_type", "unknown"))
    
    logger.info(f"📋 Создаем ClearML задачу: {task_name}")
    logger.info(f"🏷️  Теги: {tags}")
    
    # Настраиваем ClearML окружение
    setup_clearml_environment()
    
    # Создаем ClearML задачу
    task = create_clearml_task(
        project_name=project_name,
        task_name=task_name,
        tags=tags
    )
    
    # Получаем логгер
    clearml_logger = Logger.current_logger()
    
    # Загружаем конфигурацию
    if "config" in data:
        logger.info("📋 Загружаем конфигурацию...")
        task.connect(data["config"])
    
    # Загружаем текстовые логи
    if "text_logs" in data:
        logger.info("📝 Загружаем текстовые логи...")
        for log_entry in data["text_logs"]:
            clearml_logger.report_text(log_entry["text"])
    
    # Загружаем скалярные метрики
    if "scalar_metrics" in data:
        logger.info("📊 Загружаем скалярные метрики...")
        for title, series_dict in data["scalar_metrics"].items():
            for series, values in series_dict.items():
                for entry in values:
                    clearml_logger.report_scalar(
                        title=title,
                        series=series,
                        value=entry["value"],
                        iteration=entry["iteration"]
                    )
    
    # Загружаем одиночные значения
    if "single_values" in data:
        logger.info("📈 Загружаем одиночные значения...")
        for name, entry in data["single_values"].items():
            clearml_logger.report_single_value(name, entry["value"])
    
    # Загружаем таблицы
    if "tables" in data:
        logger.info("📋 Загружаем таблицы...")
        for table_entry in data["tables"]:
            df = pd.DataFrame(table_entry["data"])
            clearml_logger.report_table(
                title=table_entry["title"],
                series=table_entry["series"],
                table_plot=df,
                iteration=table_entry["iteration"]
            )
    
    # Загружаем артефакты
    if "artifacts" in data:
        logger.info("📦 Загружаем артефакты...")
        for artifact_meta in data["artifacts"]:
            artifact_path = Path(artifact_meta["saved_path"])
            if artifact_path.exists():
                task.upload_artifact(
                    name=artifact_meta["name"],
                    artifact_object=str(artifact_path),
                    metadata=artifact_meta
                )
                logger.info(f"  ✅ Загружен артефакт: {artifact_meta['name']}")
            else:
                logger.warning(f"  ⚠️  Артефакт не найден: {artifact_path}")
    
    logger.info("✅ Все данные успешно загружены в ClearML!")
    logger.info(f"🔗 Задача: {task.get_output_log_web_page()}")


def main():
    parser = argparse.ArgumentParser(
        description='Загрузка локальных результатов эксперимента в ClearML'
    )
    parser.add_argument(
        'results_dir',
        type=str,
        help='Директория с результатами эксперимента'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='slm-experiments',
        help='Название проекта в ClearML (по умолчанию: slm-experiments)'
    )
    parser.add_argument(
        '--task-name',
        type=str,
        default=None,
        help='Название задачи (по умолчанию берется из metadata)'
    )
    parser.add_argument(
        '--tags',
        type=str,
        nargs='+',
        default=None,
        help='Теги для задачи (по умолчанию берутся из конфигурации)'
    )
    parser.add_argument(
        '--env-file',
        type=str,
        default='.env',
        help='Путь к .env файлу с настройками ClearML'
    )
    
    args = parser.parse_args()
    
    # Загружаем .env файл
    env_path = Path(args.env_file)
    if env_path.exists():
        load_dotenv(env_path)
    else:
        logger.warning(f"⚠️  Файл .env не найден: {env_path}")
    
    try:
        upload_to_clearml(
            results_dir=Path(args.results_dir),
            project_name=args.project,
            task_name=args.task_name,
            tags=args.tags
        )
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке результатов: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

