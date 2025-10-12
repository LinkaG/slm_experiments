#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для скачивания данных из S3 хранилища.

Поддерживает скачивание:
- Natural Questions (NQ) данных
- SimpleQA данных
- Произвольных файлов из S3

Использование:
    poetry run python download_from_s3.py --help
"""

import boto3
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dotenv import load_dotenv
import os

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
    
    # Проверяем обязательные параметры
    required = ['aws_access_key_id', 'aws_secret_access_key', 'bucket']
    missing = [key for key in required if not config[key]]
    
    if missing:
        raise ValueError(f"Отсутствуют обязательные параметры в .env: {missing}")
    
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

def download_file_from_s3(
    s3_client: boto3.client,
    bucket: str,
    s3_key: str,
    local_file: Path,
    overwrite: bool = False
) -> bool:
    """
    Скачивает файл из S3.
    
    Args:
        s3_client: S3 клиент
        bucket: Имя bucket
        s3_key: S3 ключ (путь в bucket)
        local_file: Локальный путь для сохранения
        overwrite: Перезаписывать существующий файл
        
    Returns:
        bool: True если успешно, False если ошибка
    """
    # Проверяем существование локального файла
    if local_file.exists() and not overwrite:
        logger.warning(f"Файл уже существует: {local_file}. Используйте --overwrite для перезаписи.")
        return False
    
    # Создаем директорию если не существует
    local_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Скачиваем s3://{bucket}/{s3_key} -> {local_file}")
        
        s3_client.download_file(bucket, s3_key, str(local_file))
        
        logger.info(f"✅ Успешно скачан: {local_file}")
        return True
        
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"❌ Файл не найден в S3: s3://{bucket}/{s3_key}")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка скачивания s3://{bucket}/{s3_key}: {e}")
        return False

def list_s3_files(s3_client: boto3.client, bucket: str, prefix: str = "") -> List[str]:
    """Получает список файлов в S3 bucket с указанным префиксом."""
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        files = []
        
        if 'Contents' in response:
            for obj in response['Contents']:
                files.append(obj['Key'])
        
        return files
    except Exception as e:
        logger.error(f"Ошибка получения списка файлов: {e}")
        return []

def download_nq_data(s3_client: boto3.client, bucket: str, data_dir: Path, overwrite: bool = False):
    """Скачивает NQ данные из S3."""
    logger.info("📊 Скачиваем Natural Questions данные...")
    
    # Определяем файлы для скачивания
    nq_files = [
        ('NQ-open.dev.merged.jsonl', 'NQ-open.dev.merged.jsonl'),      # Исходный файл
        ('nq_full_dataset.json', 'nq_full_dataset.json'),              # Конвертированный
        ('nq_converted_eval.json', 'nq_converted_eval.json'),          # Eval данные
        ('nq_converted_train.json', 'nq_converted_train.json')          # Train данные
    ]
    
    success_count = 0
    for s3_key, local_path in nq_files:
        # NQ файлы сохраняем в data/nq/
        local_file = data_dir / 'nq' / local_path
        if download_file_from_s3(s3_client, bucket, s3_key, local_file, overwrite):
            success_count += 1
    
    logger.info(f"📈 Скачано {success_count} NQ файлов")

def download_simple_qa_data(s3_client: boto3.client, bucket: str, data_dir: Path, overwrite: bool = False):
    """Скачивает SimpleQA данные из S3."""
    logger.info("📊 Скачиваем SimpleQA данные...")
    
    # Определяем файлы для скачивания
    simple_qa_files = [
        ('simple_qa_test_set_with_documents.csv', 'simple_qa_test_set_with_documents.csv'),  # Исходный файл
        ('simple_qa_converted.json', 'simple_qa_converted.json'),                              # Конвертированный
        ('simple_qa_train.json', 'simple_qa_train.json'),                                     # Train данные
        ('simple_qa_eval.json', 'simple_qa_eval.json')                                         # Eval данные
    ]
    
    success_count = 0
    for s3_key, local_path in simple_qa_files:
        # SimpleQA файлы сохраняем в data/simple_qa/
        local_file = data_dir / 'simple_qa' / local_path
        if download_file_from_s3(s3_client, bucket, s3_key, local_file, overwrite):
            success_count += 1
    
    logger.info(f"📈 Скачано {success_count} SimpleQA файлов")

def download_custom_file(s3_client: boto3.client, bucket: str, s3_key: str, local_file: Path, overwrite: bool = False):
    """Скачивает произвольный файл из S3."""
    logger.info(f"📥 Скачиваем произвольный файл: s3://{bucket}/{s3_key}")
    
    if download_file_from_s3(s3_client, bucket, s3_key, local_file, overwrite):
        logger.info(f"✅ Файл успешно скачан: {local_file}")
    else:
        logger.error(f"❌ Ошибка скачивания файла: s3://{bucket}/{s3_key}")

def browse_s3_bucket(s3_client: boto3.client, bucket: str, prefix: str = ""):
    """Просматривает содержимое S3 bucket."""
    logger.info(f"🔍 Просматриваем содержимое bucket: {bucket}")
    if prefix:
        logger.info(f"📁 Префикс: {prefix}")
    
    files = list_s3_files(s3_client, bucket, prefix)
    
    if files:
        logger.info(f"📋 Найдено {len(files)} файлов:")
        for file in files:
            logger.info(f"  - {file}")
    else:
        logger.info("📭 Файлы не найдены")

def test_s3_connection(s3_client: boto3.client, bucket: str) -> bool:
    """Тестирует подключение к S3."""
    try:
        logger.info(f"🔍 Тестируем подключение к S3 bucket: {bucket}")
        s3_client.head_bucket(Bucket=bucket)
        logger.info("✅ Подключение к S3 успешно!")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к S3: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Скачивание данных из S3 хранилища')
    parser.add_argument('--mode', choices=['nq', 'simple_qa', 'custom', 'browse', 'test'], required=True,
                       help='Режим скачивания: nq, simple_qa, custom, browse, test')
    parser.add_argument('--data-dir', default='data',
                       help='Директория для сохранения данных (по умолчанию: data)')
    parser.add_argument('--s3-key',
                       help='S3 ключ для скачивания (для режима custom)')
    parser.add_argument('--local-file',
                       help='Локальный путь для сохранения (для режима custom)')
    parser.add_argument('--prefix',
                       help='Префикс для поиска файлов (для режима browse)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Перезаписывать существующие файлы')
    parser.add_argument('--bucket',
                       help='S3 bucket (переопределяет .env)')
    
    args = parser.parse_args()
    
    try:
        # Загружаем конфигурацию S3
        config = load_s3_config()
        
        # Переопределяем bucket если указан
        if args.bucket:
            config['bucket'] = args.bucket
        
        # Создаем S3 клиент
        s3_client = create_s3_client(config)
        bucket = config['bucket']
        
        # Тестируем подключение
        if not test_s3_connection(s3_client, bucket):
            return 1
        
        data_dir = Path(args.data_dir)
        
        if args.mode == 'test':
            logger.info("✅ Тест подключения завершен успешно!")
            return 0
        
        elif args.mode == 'browse':
            browse_s3_bucket(s3_client, bucket, args.prefix or "")
        
        elif args.mode == 'nq':
            download_nq_data(s3_client, bucket, data_dir, args.overwrite)
        
        elif args.mode == 'simple_qa':
            download_simple_qa_data(s3_client, bucket, data_dir, args.overwrite)
        
        elif args.mode == 'custom':
            if not args.s3_key or not args.local_file:
                logger.error("Для режима custom необходимо указать --s3-key и --local-file")
                return 1
            
            local_file = Path(args.local_file)
            download_custom_file(s3_client, bucket, args.s3_key, local_file, args.overwrite)
        
        logger.info("🎉 Скачивание завершено!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
