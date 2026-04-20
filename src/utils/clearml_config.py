"""
Утилиты для настройки ClearML с использованием переменных окружения из .env файла
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


def _ensure_clearml_conf_file() -> None:
    """
    Кладёт clearml.conf.docker в ~/.clearml.conf до import clearml.

    В контейнере с HOME=/tmp битый или пустой /tmp/.clearml.conf остаётся между
    попытками в рамках одного run; без sdk.storage.direct_access boto3 ходит в AWS
    и даёт InvalidAccessKeyId для minioadmin. В Docker всегда перезаписываем из репозитория.
    """
    import shutil

    project_root = Path(__file__).resolve().parent.parent.parent
    docker_cfg = project_root / "clearml.conf.docker"
    if not docker_cfg.is_file():
        return
    target = Path.home() / ".clearml.conf"
    in_docker = Path("/.dockerenv").exists()
    if in_docker or not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(docker_cfg, target)


_ensure_clearml_conf_file()

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
    
    env_path = Path(env_file)
    if not env_path.exists():
        # Если .env не найден, возвращаем пустой словарь (работа без ClearML)
        return {}
    
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
    
    if not config:
        # Если конфигурация пустая (нет .env или нет параметров), ничего не делаем
        return
    
    # Устанавливаем переменные окружения для ClearML
    for key, value in config.items():
        os.environ[key] = value


def create_clearml_task(
    project_name: str = "no_context",
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

    _ensure_clearml_conf_file()
    config_path = Path.home() / ".clearml.conf"
    
    # Парсим конфигурационный файл и устанавливаем переменные окружения
    # ClearML SDK может не читать файл правильно, поэтому используем переменные окружения
    if config_path.exists():
        try:
            import re
            with open(config_path, 'r') as f:
                content = f.read()
                
                # Извлекаем значения из HOCON конфигурации
                # Ищем api_server, web_server, files_server
                api_match = re.search(r'api_server:\s*([^\s\n]+)', content)
                web_match = re.search(r'web_server:\s*([^\s\n]+)', content)
                files_match = re.search(r'files_server:\s*([^\s\n]+)', content)
                
                if api_match:
                    os.environ['CLEARML_API_HOST'] = api_match.group(1)
                if web_match:
                    os.environ['CLEARML_WEB_HOST'] = web_match.group(1)
                if files_match:
                    os.environ['CLEARML_FILES_HOST'] = files_match.group(1)
                
                # Ищем credentials (access_key и secret_key)
                access_key_match = re.search(r'"access_key"\s*=\s*"([^"]+)"', content)
                secret_key_match = re.search(r'"secret_key"\s*=\s*"([^"]+)"', content)
                
                if access_key_match:
                    os.environ['CLEARML_API_ACCESS_KEY'] = access_key_match.group(1)
                if secret_key_match:
                    os.environ['CLEARML_API_SECRET_KEY'] = secret_key_match.group(1)
        except Exception as e:
            # Если не удалось распарсить, продолжаем без переменных окружения
            import logging
            logging.getLogger(__name__).warning(f"Не удалось распарсить конфигурацию: {e}")
    
    # output_uri=False: не выставлять default_output_uri (s3://...) и не вызывать
    # check_write_permissions — у SDK при s3:// без корректного endpoint часто уходит запрос на AWS.
    # Артефакты в MinIO загружаются в ExperimentRunner через boto3 и регистрируются в задаче отдельно.
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
        tags=tags or [],
        auto_connect_frameworks=False,
        output_uri=False,
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


def log_prompt_template(logger: Logger, prompt_template_no_context: str, 
                        prompt_template_with_context: str, use_context: bool) -> None:
    """
    Логирует шаблон промпта, используемый в эксперименте
    
    Args:
        logger: ClearML Logger
        prompt_template_no_context: Шаблон промпта без контекста
        prompt_template_with_context: Шаблон промпта с контекстом
        use_context: Используется ли контекст в текущем эксперименте
    """
    logger.report_text("📝 Шаблон промпта:")
    logger.report_text("")
    
    if use_context:
        logger.report_text("```")
        logger.report_text(prompt_template_with_context)
        logger.report_text("```")
    else:
        logger.report_text("```")
        logger.report_text(prompt_template_no_context)
        logger.report_text("```")


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


def log_metrics_to_clearml(logger: Logger, metrics: Dict[str, Any]) -> None:
    """
    Логирует метрики в ClearML как таблицу и скаляры
    
    Args:
        logger: ClearML Logger
        metrics: Словарь с метриками
    """
    # Функция для форматирования значений
    def format_metric_value(key: str, value) -> str:
        """Форматирует значение метрики в зависимости от её типа."""
        if not isinstance(value, (int, float)):
            return str(value)
        
        # Байты - целые числа
        if 'bytes' in key.lower():
            return f"{int(value):,}"
        # MB - 2 знака после запятой
        elif 'mb' in key.lower() or 'ram' in key.lower():
            return f"{value:.2f}"
        # Recall и другие метрики качества - 4 знака
        elif 'recall' in key.lower() or 'precision' in key.lower() or 'f1' in key.lower():
            return f"{value:.4f}"
        # Время - 2 знака
        elif 'time' in key.lower() or 'duration' in key.lower() or 'seconds' in key.lower():
            return f"{value:.2f}"
        # Счетчики - целые числа
        elif 'num' in key.lower() or 'count' in key.lower() or 'examples' in key.lower():
            return f"{int(value)}"
        # По умолчанию - 4 знака
        else:
            return f"{value:.4f}" if isinstance(value, float) else str(value)
    
    # Логируем в текстовом виде
    logger.report_text("📊 Финальные результаты эксперимента:")
    logger.report_text("")
    
    # Группируем метрики по категориям
    quality_metrics = {}
    memory_metrics = {}
    size_metrics = {}
    other_metrics = {}
    
    for key, value in metrics.items():
        key_lower = key.lower()
        if 'recall' in key_lower or 'precision' in key_lower or 'f1' in key_lower:
            quality_metrics[key] = value
        elif 'ram' in key_lower or 'memory' in key_lower:
            memory_metrics[key] = value
        elif 'size' in key_lower:
            size_metrics[key] = value
        else:
            other_metrics[key] = value
    
    # Логируем метрики качества
    if quality_metrics:
        logger.report_text("🎯 Метрики качества:")
        for k, v in quality_metrics.items():
            logger.report_text(f"  {k}: {format_metric_value(k, v)}")
        logger.report_text("")
    
    # Логируем размеры модели и индекса
    if size_metrics:
        logger.report_text("📦 Размеры модели и индекса:")
        for k, v in size_metrics.items():
            logger.report_text(f"  {k}: {format_metric_value(k, v)}")
        logger.report_text("")
    
    # Логируем использование памяти
    if memory_metrics:
        logger.report_text("💾 Пиковое использование памяти:")
        for k, v in memory_metrics.items():
            logger.report_text(f"  {k}: {format_metric_value(k, v)} MB")
        logger.report_text("")
    
    # Логируем остальные метрики
    if other_metrics:
        logger.report_text("📋 Прочие метрики:")
        for k, v in other_metrics.items():
            logger.report_text(f"  {k}: {format_metric_value(k, v)}")
    
    # Создаем таблицу для вкладки PLOTS
    try:
        import pandas as pd
        
        # Создаем DataFrame с метриками (красиво отформатированный)
        df = pd.DataFrame([{
            'Metric': k,
            'Value': format_metric_value(k, v)
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
        if isinstance(value, (int, float)):
            logger.report_single_value(
                name=metric_name,
                value=value
            )
