#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для настройки учетных данных ClearML.
Создает конфигурационный файл с правильными настройками для удаленного сервера.
"""

import os
import sys
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_clearml_config():
    """Создает конфигурационный файл ClearML."""
    
    config_content = """api {
    # Notice: 'host' is the api server (default port 8008), not the web server.
    api_server: http://51.250.43.3:8008
    web_server: http://51.250.43.3:8080
    files_server: http://51.250.43.3:8081
    # Credentials are generated using the webapp, http://51.250.43.3:8080
    credentials {
        "access_key" = "your-access-key"
        "secret_key" = "your-secret-key"
    }
}
sdk {
    # Storage for output models and other artifacts
    storage {
        cache {
            # Defaults to system temp folder / cache
            default_base_dir: "~/clearml/cache"
        }
        # S3 storage configuration for artifacts
        s3 {
            endpoint_url: "http://51.250.43.3:9000"
            bucket: "clearml-artifacts"
            access_key: "minio_admin_2024"
            secret_key: "Kx9mP7$vL2@nQ8!wE5&rT3*yU6+iO1-pA4^sD9~fG0"
            region: "us-east-1"
            path_style: true
            verify_ssl: false
        }
    }
}
"""
    
    # Путь к конфигурационному файлу
    config_path = Path.home() / ".clearml.conf"
    
    try:
        # Создаем директорию если не существует
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Записываем конфигурацию
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"✅ Конфигурационный файл создан: {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания конфигурации: {e}")
        return False

def setup_environment_variables():
    """Настраивает переменные окружения для ClearML."""
    
    env_vars = {
        "CLEARML_API_HOST": "http://51.250.43.3:8008",
        "CLEARML_WEB_HOST": "http://51.250.43.3:8080",
        "CLEARML_FILES_HOST": "http://51.250.43.3:8081",
        "CLEARML_S3_ENDPOINT": "http://51.250.43.3:9000",
        "CLEARML_S3_BUCKET": "clearml-artifacts",
        "CLEARML_S3_ACCESS_KEY": "minio_admin_2024",
        "CLEARML_S3_SECRET_KEY": "Kx9mP7$vL2@nQ8!wE5&rT3*yU6+iO1-pA4^sD9~fG0"
    }
    
    logger.info("🔧 Настройка переменных окружения...")
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"  {key} = {value}")
    
    return True

def create_env_file():
    """Создает .env файл с переменными окружения."""
    
    env_content = """# ClearML Configuration
CLEARML_API_HOST=http://51.250.43.3:8008
CLEARML_WEB_HOST=http://51.250.43.3:8080
CLEARML_FILES_HOST=http://51.250.43.3:8081

# S3 Storage Configuration
CLEARML_S3_ENDPOINT=http://51.250.43.3:9000
CLEARML_S3_BUCKET=clearml-artifacts
CLEARML_S3_ACCESS_KEY=minio_admin_2024
CLEARML_S3_SECRET_KEY=Kx9mP7$vL2@nQ8!wE5&rT3*yU6+iO1-pA4^sD9~fG0
CLEARML_S3_REGION=us-east-1
CLEARML_S3_PATH_STYLE=true
CLEARML_S3_VERIFY_SSL=false
"""
    
    env_path = Path(".env")
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        logger.info(f"✅ .env файл создан: {env_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания .env файла: {e}")
        return False

def print_instructions():
    """Выводит инструкции по настройке."""
    
    instructions = """
📋 ИНСТРУКЦИИ ПО НАСТРОЙКЕ CLEARML:

1. 🔑 Получение учетных данных:
   - Откройте веб-интерфейс: http://51.250.43.3:8080
   - Зарегистрируйтесь или войдите в систему
   - Перейдите в Settings → Workspace → Create new credentials
   - Скопируйте access_key и secret_key

2. ⚙️ Обновление конфигурации:
   - Откройте файл ~/.clearml.conf
   - Замените "your-access-key" и "your-secret-key" на ваши учетные данные
   - Сохраните файл

3. 🧪 Тестирование подключения:
   python test_clearml_connection.py

4. 🚀 Запуск эксперимента:
   poetry run python -m src.cli run-experiment model=smollm2_135m dataset=local_nq

📊 Доступ к результатам:
   - Веб-интерфейс: http://51.250.43.3:8080
   - API: http://51.250.43.3:8008
   - S3 хранилище: http://51.250.43.3:9000
"""
    
    logger.info(instructions)

def main():
    """Основная функция настройки."""
    logger.info("🔧 Настройка ClearML для удаленного сервера...")
    logger.info("=" * 60)
    
    results = []
    
    # Настройка переменных окружения
    logger.info("1. Настройка переменных окружения...")
    results.append(("Environment variables", setup_environment_variables()))
    
    # Создание конфигурационного файла
    logger.info("2. Создание конфигурационного файла...")
    results.append(("Config file", create_clearml_config()))
    
    # Создание .env файла
    logger.info("3. Создание .env файла...")
    results.append(("Env file", create_env_file()))
    
    # Результаты
    logger.info("=" * 60)
    logger.info("📊 РЕЗУЛЬТАТЫ НАСТРОЙКИ:")
    
    passed = 0
    total = len(results)
    
    for step_name, result in results:
        status = "✅ УСПЕШНО" if result else "❌ ОШИБКА"
        logger.info(f"  {step_name}: {status}")
        if result:
            passed += 1
    
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("🎉 Настройка завершена успешно!")
        print_instructions()
        return 0
    else:
        logger.warning("⚠️ Некоторые шаги завершились с ошибками.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
