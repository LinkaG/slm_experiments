"""
Утилиты для настройки ClearML с использованием переменных окружения из .env файла
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from clearml import Task, Logger


def load_clearml_config(env_file: Optional[str] = None) -> Dict[str, str]:
    """
    Загружает конфигурацию ClearML из .env файла
    
    Args:
        env_file: Путь к .env файлу (по умолчанию .env в корне проекта)
    
    Returns:
        Словарь с настройками ClearML
    """
    if env_file is None:
        # Ищем .env файл в корне проекта
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"
    
    if not Path(env_file).exists():
        raise FileNotFoundError(f"Файл .env не найден: {env_file}")
    
    # Загружаем переменные окружения
    load_dotenv(env_file)
    
    # Извлекаем настройки ClearML
    config = {}
    
    # Основные настройки ClearML
    clearml_vars = [
        'CLEARML_API_HOST',
        'CLEARML_WEB_HOST', 
        'CLEARML_FILES_HOST',
        'CLEARML_S3_ENDPOINT',
        'CLEARML_S3_BUCKET',
        'CLEARML_S3_ACCESS_KEY',
        'CLEARML_S3_SECRET_KEY',
        'CLEARML_S3_REGION',
        'CLEARML_S3_PATH_STYLE',
        'CLEARML_S3_VERIFY_SSL'
    ]
    
    for var in clearml_vars:
        value = os.getenv(var)
        if value is not None:
            config[var] = value
    
    return config


def setup_clearml_environment(env_file: Optional[str] = None) -> None:
    """
    Настраивает переменные окружения для ClearML
    
    Args:
        env_file: Путь к .env файлу
    """
    config = load_clearml_config(env_file)
    
    # Устанавливаем переменные окружения для ClearML
    for key, value in config.items():
        os.environ[key] = value
    
    # Дополнительные настройки для S3
    if 'CLEARML_S3_ENDPOINT' in config:
        os.environ['CLEARML_S3_ENDPOINT'] = config['CLEARML_S3_ENDPOINT']
    if 'CLEARML_S3_BUCKET' in config:
        os.environ['CLEARML_S3_BUCKET'] = config['CLEARML_S3_BUCKET']
    if 'CLEARML_S3_ACCESS_KEY' in config:
        os.environ['CLEARML_S3_ACCESS_KEY'] = config['CLEARML_S3_ACCESS_KEY']
    if 'CLEARML_S3_SECRET_KEY' in config:
        os.environ['CLEARML_S3_SECRET_KEY'] = config['CLEARML_S3_SECRET_KEY']
    if 'CLEARML_S3_REGION' in config:
        os.environ['CLEARML_S3_REGION'] = config['CLEARML_S3_REGION']
    if 'CLEARML_S3_PATH_STYLE' in config:
        os.environ['CLEARML_S3_PATH_STYLE'] = config['CLEARML_S3_PATH_STYLE']
    if 'CLEARML_S3_VERIFY_SSL' in config:
        os.environ['CLEARML_S3_VERIFY_SSL'] = config['CLEARML_S3_VERIFY_SSL']


def create_clearml_task(
    project_name: str = "slm-experiments",
    task_name: str = "experiment",
    tags: Optional[list] = None,
    env_file: Optional[str] = None
) -> Task:
    """
    Создает и настраивает ClearML Task с использованием конфигурации из .env
    
    Args:
        project_name: Название проекта в ClearML
        task_name: Название задачи
        tags: Список тегов для задачи
        env_file: Путь к .env файлу
    
    Returns:
        Настроенный ClearML Task
    """
    # Настраиваем окружение
    setup_clearml_environment(env_file)
    
    # Создаем задачу
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        tags=tags or [],
        auto_connect_frameworks=False  # Отключаем автоматическое подключение фреймворков
    )
    
    return task


def get_clearml_logger() -> Logger:
    """
    Получает настроенный ClearML Logger
    
    Returns:
        ClearML Logger
    """
    return Logger.current_logger()


def log_experiment_config(logger: Logger, config: Dict[str, Any]) -> None:
    """
    Логирует полную конфигурацию эксперимента в ClearML
    
    Args:
        logger: ClearML Logger
        config: Конфигурация эксперимента
    """
    # Логируем основную информацию
    logger.report_text("🚀 Начало эксперимента")
    logger.report_text(f"📁 Директория результатов: {config.get('output_dir', 'unknown')}")
    logger.report_text(f"🤖 Модель: {config.get('model', {}).get('name', 'unknown')}")
    logger.report_text(f"📊 Датасет: {config.get('dataset', {}).get('name', 'unknown')}")
    logger.report_text(f"🔍 Режим: {config.get('experiment_mode', {}).get('name', 'unknown')}")
    
    # Логируем детальную конфигурацию
    logger.report_text("📋 Полная конфигурация эксперимента:")
    for section, values in config.items():
        if isinstance(values, dict):
            logger.report_text(f"  {section}:")
            for key, value in values.items():
                logger.report_text(f"    {key}: {value}")
        else:
            logger.report_text(f"  {section}: {values}")


def log_predictions_to_clearml(logger: Logger, predictions: list, max_examples: int = 100) -> None:
    """
    Логирует предсказания модели в ClearML
    
    Args:
        logger: ClearML Logger
        predictions: Список предсказаний (PredictionItem objects or dicts)
        max_examples: Максимальное количество примеров для логирования
    """
    logger.report_text(f"📝 Примеры предсказаний модели (первые {min(len(predictions), max_examples)} из {len(predictions)}):")
    
    for i, pred in enumerate(predictions[:max_examples]):
        logger.report_text(f"\n--- Пример {i+1} ---")
        
        # Поддержка как объектов, так и словарей
        if hasattr(pred, 'question'):
            # Это объект PredictionItem
            logger.report_text(f"Вопрос: {pred.question}")
            logger.report_text(f"Предсказанный ответ: {pred.predicted_answer}")
            logger.report_text(f"Правильный ответ: {pred.ground_truth}")
            logger.report_text(f"Token Recall: {pred.token_recall:.4f}")
            
            if pred.contexts:
                logger.report_text(f"Контекст: {pred.contexts}")
        else:
            # Это словарь
            logger.report_text(f"Вопрос: {pred.get('question', 'N/A')}")
            logger.report_text(f"Предсказанный ответ: {pred.get('predicted_answer', 'N/A')}")
            logger.report_text(f"Правильный ответ: {pred.get('ground_truth', 'N/A')}")
            logger.report_text(f"Token Recall: {pred.get('token_recall', 0.0):.4f}")
            
            if pred.get('contexts'):
                logger.report_text(f"Контекст: {pred.get('contexts', [])}")
    
    if len(predictions) > max_examples:
        logger.report_text(f"\n... и еще {len(predictions) - max_examples} примеров")


def log_metrics_to_clearml(logger: Logger, metrics: Dict[str, float]) -> None:
    """
    Логирует метрики в ClearML как таблицу и скаляры
    
    Args:
        logger: ClearML Logger
        metrics: Словарь с метриками
    """
    # Логируем в текстовом виде
    logger.report_text("📊 Финальные результаты эксперимента:")
    
    for metric_name, value in metrics.items():
        logger.report_text(f"  {metric_name}: {value:.4f}")
    
    # Создаем таблицу для вкладки PLOTS
    try:
        import pandas as pd
        
        # Создаем DataFrame с метриками
        df = pd.DataFrame([{
            'Metric': k,
            'Value': f"{v:.4f}" if isinstance(v, float) else str(v)
        } for k, v in metrics.items()])
        
        # Логируем как таблицу в PLOTS
        logger.report_table(
            title="Final Metrics",
            series="Summary",
            table_plot=df,
            iteration=0
        )
    except Exception as e:
        # Fallback если pandas недоступен
        logger.report_text(f"⚠️  Could not create metrics table: {e}")
    
    # Логируем каждую метрику как single value scalar (без графика)
    for metric_name, value in metrics.items():
        logger.report_single_value(
            name=metric_name,
            value=value
        )
