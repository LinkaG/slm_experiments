#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования подключения к ClearML серверу.
Проверяет доступность API, веб-интерфейса и S3 хранилища.

Использование:
    # Локальный запуск (на хосте)
    poetry run python test_clearml_connection.py
    
    # Запуск через Docker сеть
    ./run_in_docker_network.sh test_clearml_connection.py
"""

import os
import sys
import requests
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_api_connection():
    """Тестирует подключение к ClearML API."""
    api_url = "http://localhost:8008"
    try:
        response = requests.get(f"{api_url}/v2.3/system/version", timeout=10)
        if response.status_code == 200:
            logger.info(f"✅ ClearML API доступен: {api_url}")
            return True
        else:
            logger.error(f"❌ ClearML API недоступен: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к ClearML API: {e}")
        return False

def test_web_connection():
    """Тестирует подключение к ClearML Web UI."""
    web_url = "http://localhost:8080"
    try:
        response = requests.get(web_url, timeout=10)
        if response.status_code == 200:
            logger.info(f"✅ ClearML Web UI доступен: {web_url}")
            return True
        else:
            logger.error(f"❌ ClearML Web UI недоступен: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к ClearML Web UI: {e}")
        return False

def test_s3_connection():
    """Тестирует подключение к MinIO S3."""
    s3_url = "http://localhost:9000"
    try:
        response = requests.get(s3_url, timeout=10)
        if response.status_code in [200, 403]:  # 403 означает что сервер доступен, но нужна авторизация
            logger.info(f"✅ MinIO S3 доступен: {s3_url}")
            return True
        else:
            logger.error(f"❌ MinIO S3 недоступен: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к MinIO S3: {e}")
        return False

def test_clearml_import():
    """Тестирует импорт и инициализацию ClearML."""
    try:
        from clearml import Task, Logger
        logger.info("✅ ClearML модуль успешно импортирован")
        
        # Попытка создать тестовую задачу
        task = Task.init(
            project_name="test-connection",
            task_name="clearml-connection-test",
            auto_connect_frameworks=False
        )
        logger.info("✅ ClearML Task успешно создан")
        
        # Тест логирования
        logger_clearml = Logger.current_logger()
        if logger_clearml:
            logger_clearml.report_scalar(
                title="test",
                series="connection",
                value=1.0,
                iteration=0
            )
            logger.info("✅ ClearML Logger работает")
        
        # Закрываем задачу
        task.close()
        logger.info("✅ ClearML Task успешно закрыт")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при работе с ClearML: {e}")
        return False

def check_config_file():
    """Проверяет наличие и корректность конфигурационного файла."""
    config_path = Path.home() / ".clearml.conf"
    if config_path.exists():
        logger.info(f"✅ Конфигурационный файл найден: {config_path}")
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                if "localhost" in content:
                    logger.info("✅ Конфигурация содержит правильный IP адрес")
                    return True
                else:
                    logger.warning("⚠️ Конфигурация может быть устаревшей")
                    return False
        except Exception as e:
            logger.error(f"❌ Ошибка чтения конфигурации: {e}")
            return False
    else:
        logger.warning(f"⚠️ Конфигурационный файл не найден: {config_path}")
        logger.info("💡 Скопируйте clearml.conf в ~/.clearml.conf")
        return False

def main():
    """Основная функция тестирования."""
    logger.info("🔍 Тестирование подключения к ClearML серверу...")
    logger.info("=" * 60)
    
    results = []
    
    # Тест 1: Проверка конфигурации
    logger.info("1. Проверка конфигурационного файла...")
    results.append(("Config file", check_config_file()))
    
    # Тест 2: Подключение к API
    logger.info("2. Тестирование ClearML API...")
    results.append(("API Connection", test_api_connection()))
    
    # Тест 3: Подключение к Web UI
    logger.info("3. Тестирование ClearML Web UI...")
    results.append(("Web UI Connection", test_web_connection()))
    
    # Тест 4: Подключение к S3
    logger.info("4. Тестирование MinIO S3...")
    results.append(("S3 Connection", test_s3_connection()))
    
    # Тест 5: Импорт и работа с ClearML
    logger.info("5. Тестирование ClearML SDK...")
    results.append(("ClearML SDK", test_clearml_import()))
    
    # Результаты
    logger.info("=" * 60)
    logger.info("📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 60)
    logger.info(f"📈 Итого: {passed}/{total} тестов пройдено")
    
    if passed == total:
        logger.info("🎉 Все тесты пройдены! ClearML готов к работе.")
        return 0
    else:
        logger.warning("⚠️ Некоторые тесты провалены. Проверьте настройки.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
