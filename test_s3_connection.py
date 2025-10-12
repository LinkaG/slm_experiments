#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования подключения к S3 хранилищу.

Проверяет:
- Доступность S3 хранилища
- Корректность креденшиалов
- Доступность bucket
- Список доступных файлов

Использование:
    poetry run python test_s3_connection.py
"""

import boto3
import argparse
from pathlib import Path
from typing import Dict, Any, List
import logging
from dotenv import load_dotenv
import os
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_s3_config() -> Dict[str, str]:
    """Загружает конфигурацию S3 из .env файла."""
    load_dotenv()
    
    config = {
        'aws_access_key_id': os.getenv('CLEARML_S3_ACCESS_KEY'),
        'aws_secret_access_key': os.getenv('CLEARML_S3_SECRET_KEY'),
        'region_name': os.getenv('CLEARML_S3_REGION', 'us-east-1'),
        'endpoint_url': os.getenv('CLEARML_S3_ENDPOINT'),
        'bucket': os.getenv('CLEARML_S3_BUCKET')
    }
    
    return config

def create_s3_client(config: Dict[str, str]) -> boto3.client:
    """Создает S3 клиент с конфигурацией."""
    s3_config = {
        'aws_access_key_id': config['aws_access_key_id'],
        'aws_secret_access_key': config['aws_secret_access_key'],
        'region_name': config['region_name']
    }
    
    # Добавляем endpoint_url если указан (для MinIO и других S3-совместимых хранилищ)
    if config.get('endpoint_url'):
        s3_config['endpoint_url'] = config['endpoint_url']
    
    return boto3.client('s3', **s3_config)

def test_credentials(s3_client: boto3.client) -> bool:
    """Тестирует креденшиалы S3."""
    try:
        logger.info("🔐 Тестируем креденшиалы...")
        s3_client.list_buckets()
        logger.info("✅ Креденшиалы корректны!")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка креденшиалов: {e}")
        return False

def test_bucket_access(s3_client: boto3.client, bucket: str) -> bool:
    """Тестирует доступ к bucket."""
    try:
        logger.info(f"🪣 Тестируем доступ к bucket: {bucket}")
        s3_client.head_bucket(Bucket=bucket)
        logger.info("✅ Доступ к bucket успешен!")
        return True
    except s3_client.exceptions.NoSuchBucket:
        logger.error(f"❌ Bucket не найден: {bucket}")
        return False
    except s3_client.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '403':
            logger.error(f"❌ Нет доступа к bucket: {bucket}")
        else:
            logger.error(f"❌ Ошибка доступа к bucket: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {e}")
        return False

def list_buckets(s3_client: boto3.client) -> List[str]:
    """Получает список доступных buckets."""
    try:
        logger.info("📋 Получаем список доступных buckets...")
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        
        logger.info(f"✅ Найдено {len(buckets)} buckets:")
        for bucket in buckets:
            logger.info(f"  - {bucket}")
        
        return buckets
    except Exception as e:
        logger.error(f"❌ Ошибка получения списка buckets: {e}")
        return []

def list_bucket_contents(s3_client: boto3.client, bucket: str, prefix: str = "") -> List[Dict[str, Any]]:
    """Получает содержимое bucket."""
    try:
        logger.info(f"📁 Получаем содержимое bucket: {bucket}")
        if prefix:
            logger.info(f"🔍 Префикс: {prefix}")
        
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        objects = []
        
        if 'Contents' in response:
            for obj in response['Contents']:
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified']
                })
        
        logger.info(f"✅ Найдено {len(objects)} объектов")
        
        # Показываем первые 10 объектов
        for i, obj in enumerate(objects[:10]):
            size_mb = obj['size'] / (1024 * 1024)
            logger.info(f"  {i+1}. {obj['key']} ({size_mb:.2f} MB, {obj['last_modified']})")
        
        if len(objects) > 10:
            logger.info(f"  ... и еще {len(objects) - 10} объектов")
        
        return objects
    except Exception as e:
        logger.error(f"❌ Ошибка получения содержимого bucket: {e}")
        return []

def test_file_operations(s3_client: boto3.client, bucket: str) -> bool:
    """Тестирует операции с файлами (создание, чтение, удаление тестового файла)."""
    try:
        test_key = f"test_connection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        test_content = f"Test file created at {datetime.now()}"
        
        logger.info(f"📝 Тестируем операции с файлами...")
        
        # Создаем тестовый файл
        logger.info(f"  Создаем тестовый файл: {test_key}")
        s3_client.put_object(
            Bucket=bucket,
            Key=test_key,
            Body=test_content.encode('utf-8')
        )
        
        # Читаем тестовый файл
        logger.info(f"  Читаем тестовый файл: {test_key}")
        response = s3_client.get_object(Bucket=bucket, Key=test_key)
        content = response['Body'].read().decode('utf-8')
        
        if content == test_content:
            logger.info("✅ Чтение файла успешно!")
        else:
            logger.error("❌ Содержимое файла не совпадает!")
            return False
        
        # Удаляем тестовый файл
        logger.info(f"  Удаляем тестовый файл: {test_key}")
        s3_client.delete_object(Bucket=bucket, Key=test_key)
        logger.info("✅ Удаление файла успешно!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования операций с файлами: {e}")
        return False

def check_dataset_files(s3_client: boto3.client, bucket: str) -> Dict[str, bool]:
    """Проверяет наличие файлов датасетов."""
    logger.info("📊 Проверяем наличие файлов датасетов...")
    
    # Определяем файлы датасетов (реальная структура в S3)
    dataset_files = {
        'NQ Original': 'NQ-open.dev.merged.jsonl',
        'NQ Full Dataset': 'nq_full_dataset.json',
        'NQ Eval': 'nq_converted_eval.json',
        'NQ Train': 'nq_converted_train.json',
        'SimpleQA Original': 'simple_qa_test_set_with_documents.csv',
        'SimpleQA Dataset': 'simple_qa_converted.json',
        'SimpleQA Train': 'simple_qa_train.json',
        'SimpleQA Eval': 'simple_qa_eval.json'
    }
    
    results = {}
    
    for name, key in dataset_files.items():
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            results[name] = True
            logger.info(f"✅ {name}: {key}")
        except s3_client.exceptions.NoSuchKey:
            results[name] = False
            logger.warning(f"❌ {name}: {key} (не найден)")
        except Exception as e:
            results[name] = False
            logger.error(f"❌ {name}: {key} (ошибка: {e})")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Тестирование подключения к S3')
    parser.add_argument('--bucket',
                       help='S3 bucket для тестирования (переопределяет .env)')
    parser.add_argument('--prefix',
                       help='Префикс для поиска файлов')
    parser.add_argument('--test-operations', action='store_true',
                       help='Тестировать операции с файлами (создание/чтение/удаление)')
    parser.add_argument('--check-datasets', action='store_true',
                       help='Проверить наличие файлов датасетов')
    
    args = parser.parse_args()
    
    try:
        # Загружаем конфигурацию S3
        config = load_s3_config()
        
        # Переопределяем bucket если указан
        if args.bucket:
            config['bucket'] = args.bucket
        
        # Проверяем обязательные параметры
        required = ['aws_access_key_id', 'aws_secret_access_key', 'bucket']
        missing = [key for key in required if not config[key]]
        
        if missing:
            logger.error(f"❌ Отсутствуют обязательные параметры в .env: {missing}")
            logger.info("💡 Убедитесь, что файл .env содержит:")
            logger.info("   CLEARML_S3_ACCESS_KEY=your_key")
            logger.info("   CLEARML_S3_SECRET_KEY=your_secret")
            logger.info("   CLEARML_S3_BUCKET=your_bucket")
            return 1
        
        # Создаем S3 клиент
        s3_client = create_s3_client(config)
        bucket = config['bucket']
        
        logger.info("🚀 Начинаем тестирование S3 подключения...")
        logger.info(f"🔧 Конфигурация:")
        logger.info(f"   Endpoint: {config.get('endpoint_url', 'default AWS')}")
        logger.info(f"   Region: {config['region_name']}")
        logger.info(f"   Bucket: {bucket}")
        
        # Тест 1: Креденшиалы
        if not test_credentials(s3_client):
            return 1
        
        # Тест 2: Доступ к bucket
        if not test_bucket_access(s3_client, bucket):
            return 1
        
        # Тест 3: Список buckets
        list_buckets(s3_client)
        
        # Тест 4: Содержимое bucket
        list_bucket_contents(s3_client, bucket, args.prefix or "")
        
        # Тест 5: Операции с файлами (опционально)
        if args.test_operations:
            if not test_file_operations(s3_client, bucket):
                return 1
        
        # Тест 6: Проверка файлов датасетов (опционально)
        if args.check_datasets:
            dataset_results = check_dataset_files(s3_client, bucket)
            available = sum(1 for v in dataset_results.values() if v)
            total = len(dataset_results)
            logger.info(f"📊 Доступно файлов датасетов: {available}/{total}")
        
        logger.info("🎉 Все тесты завершены успешно!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
