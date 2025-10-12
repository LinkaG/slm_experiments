#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки интеграции ClearML
"""

import logging
from pathlib import Path
from src.utils.clearml_config import (
    load_clearml_config,
    setup_clearml_environment,
    create_clearml_task,
    get_clearml_logger,
    log_experiment_config,
    log_metrics_to_clearml
)

def test_clearml_config():
    """Тестирует загрузку конфигурации ClearML из .env файла."""
    print("🔧 Тестирование загрузки конфигурации ClearML...")
    
    try:
        config = load_clearml_config()
        print("✅ Конфигурация ClearML загружена успешно")
        print(f"📊 Найдено настроек: {len(config)}")
        
        for key, value in config.items():
            # Скрываем секретные ключи
            if 'SECRET' in key or 'KEY' in key:
                print(f"  {key}: {'*' * len(str(value))}")
            else:
                print(f"  {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Ошибка при загрузке конфигурации: {e}")
        return False

def test_clearml_environment():
    """Тестирует настройку окружения ClearML."""
    print("\n🌍 Тестирование настройки окружения ClearML...")
    
    try:
        setup_clearml_environment()
        print("✅ Окружение ClearML настроено успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка при настройке окружения: {e}")
        return False

def test_clearml_task_creation():
    """Тестирует создание ClearML задачи."""
    print("\n📋 Тестирование создания ClearML задачи...")
    
    try:
        task = create_clearml_task(
            project_name="slm-experiments-test",
            task_name="test-integration",
            tags=["test", "integration"]
        )
        print("✅ ClearML задача создана успешно")
        print(f"📊 ID задачи: {task.id}")
        print(f"📊 Название: {task.name}")
        print(f"📊 Проект: {task.project}")
        
        return task
    except Exception as e:
        print(f"❌ Ошибка при создании задачи: {e}")
        return None

def test_clearml_logging():
    """Тестирует логирование в ClearML."""
    print("\n📝 Тестирование логирования в ClearML...")
    
    try:
        logger = get_clearml_logger()
        
        # Тестовые данные
        test_config = {
            "model": {"name": "test-model", "size": "1.7B"},
            "dataset": {"name": "test-dataset", "samples": 1000},
            "experiment": {"name": "test-experiment", "mode": "test"}
        }
        
        # Логируем конфигурацию
        log_experiment_config(logger, test_config)
        
        # Логируем тестовые метрики
        test_metrics = {
            "token_recall": 0.85,
            "num_examples": 100,
            "duration_seconds": 120.5
        }
        log_metrics_to_clearml(logger, test_metrics)
        
        print("✅ Логирование в ClearML работает успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка при логировании: {e}")
        return False

def main():
    """Основная функция тестирования."""
    print("🚀 Запуск тестирования интеграции ClearML")
    print("=" * 50)
    
    # Тест 1: Загрузка конфигурации
    config_ok = test_clearml_config()
    
    # Тест 2: Настройка окружения
    env_ok = test_clearml_environment()
    
    # Тест 3: Создание задачи (только если предыдущие тесты прошли)
    task = None
    if config_ok and env_ok:
        task = test_clearml_task_creation()
    
    # Тест 4: Логирование (только если задача создана)
    logging_ok = False
    if task:
        logging_ok = test_clearml_logging()
    
    # Результаты
    print("\n" + "=" * 50)
    print("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"  Конфигурация: {'✅' if config_ok else '❌'}")
    print(f"  Окружение: {'✅' if env_ok else '❌'}")
    print(f"  Создание задачи: {'✅' if task else '❌'}")
    print(f"  Логирование: {'✅' if logging_ok else '❌'}")
    
    if all([config_ok, env_ok, task, logging_ok]):
        print("\n🎉 Все тесты прошли успешно! ClearML интеграция работает.")
        if task:
            print(f"🔗 Ссылка на задачу: {task.get_output_log_web_page()}")
    else:
        print("\n⚠️ Некоторые тесты не прошли. Проверьте настройки ClearML.")
    
    return all([config_ok, env_ok, task, logging_ok])

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
