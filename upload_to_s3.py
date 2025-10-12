#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для загрузки данных в S3 хранилище.

Поддерживает загрузку:
- Natural Questions (NQ) данных
- SimpleQA данных
- Произвольных JSON файлов

Использование:
    poetry run python upload_to_s3.py --help
"""

import boto3
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
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

def upload_file_to_s3(
    s3_client: boto3.client,
    bucket: str,
    local_file: Path,
    s3_key: str,
    overwrite: bool = False
) -> bool:
    """
    Загружает файл в S3.
    
    Args:
        s3_client: S3 клиент
        bucket: Имя bucket
        local_file: Локальный путь к файлу
        s3_key: S3 ключ (путь в bucket)
        overwrite: Перезаписывать существующий файл
        
    Returns:
        bool: True если успешно, False если ошибка
    """
    if not local_file.exists():
        logger.error(f"Файл не найден: {local_file}")
        return False
    
    # Проверяем существование файла в S3
    if not overwrite:
        try:
            s3_client.head_object(Bucket=bucket, Key=s3_key)
            logger.warning(f"Файл {s3_key} уже существует в S3. Используйте --overwrite для перезаписи.")
            return False
        except s3_client.exceptions.NoSuchKey:
            pass  # Файл не существует, можно загружать
    
    try:
        logger.info(f"Загружаем {local_file} -> s3://{bucket}/{s3_key}")
        
        with open(local_file, 'rb') as f:
            s3_client.upload_fileobj(f, bucket, s3_key)
        
        logger.info(f"✅ Успешно загружено: s3://{bucket}/{s3_key}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки {local_file}: {e}")
        return False

def upload_nq_data(s3_client: boto3.client, bucket: str, data_dir: Path, overwrite: bool = False):
    """Загружает NQ данные в S3."""
    logger.info("📊 Загружаем Natural Questions данные...")
    
    # Ищем файлы NQ
    nq_files = [
        ('NQ-open.dev.merged.jsonl', 'NQ-open.dev.merged.jsonl'),  # Исходный файл
        ('nq_full_dataset.json', 'nq_full_dataset.json'),          # Конвертированный
        ('nq_converted_eval.json', 'nq_converted_eval.json'),       # Eval данные
        ('nq_converted_train.json', 'nq_converted_train.json')      # Train данные
    ]
    
    success_count = 0
    for local_name, s3_key in nq_files:
        local_file = data_dir / 'nq' / local_name
        if local_file.exists():
            if upload_file_to_s3(s3_client, bucket, local_file, s3_key, overwrite):
                success_count += 1
        else:
            logger.warning(f"Файл не найден: {local_file}")
    
    logger.info(f"📈 Загружено {success_count} NQ файлов")

def upload_simple_qa_data(s3_client: boto3.client, bucket: str, data_dir: Path, overwrite: bool = False):
    """Загружает SimpleQA данные в S3."""
    logger.info("📊 Загружаем SimpleQA данные...")
    
    # Ищем файлы SimpleQA
    simple_qa_files = [
        ('simple_qa_test_set_with_documents.csv', 'simple_qa_test_set_with_documents.csv'),  # Исходный файл
        ('simple_qa_converted.json', 'simple_qa_converted.json'),                              # Конвертированный
        ('simple_qa_train.json', 'simple_qa_train.json'),                                       # Train данные
        ('simple_qa_eval.json', 'simple_qa_eval.json')                                         # Eval данные
    ]
    
    success_count = 0
    for local_name, s3_key in simple_qa_files:
        local_file = data_dir / 'simple_qa' / local_name
        if local_file.exists():
            if upload_file_to_s3(s3_client, bucket, local_file, s3_key, overwrite):
                success_count += 1
        else:
            logger.warning(f"Файл не найден: {local_file}")
    
    logger.info(f"📈 Загружено {success_count} SimpleQA файлов")

def upload_custom_file(s3_client: boto3.client, bucket: str, local_file: Path, s3_key: str, overwrite: bool = False):
    """Загружает произвольный файл в S3."""
    logger.info(f"📤 Загружаем произвольный файл: {local_file}")
    
    if upload_file_to_s3(s3_client, bucket, local_file, s3_key, overwrite):
        logger.info(f"✅ Файл успешно загружен: s3://{bucket}/{s3_key}")
    else:
        logger.error(f"❌ Ошибка загрузки файла: {local_file}")

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
    parser = argparse.ArgumentParser(description='Загрузка данных в S3 хранилище')
    parser.add_argument('--mode', choices=['nq', 'simple_qa', 'custom', 'test'], required=True,
                       help='Режим загрузки: nq, simple_qa, custom, test')
    parser.add_argument('--data-dir', default='data',
                       help='Директория с данными (по умолчанию: data)')
    parser.add_argument('--local-file', 
                       help='Локальный файл для загрузки (для режима custom)')
    parser.add_argument('--s3-key',
                       help='S3 ключ для загрузки (для режима custom)')
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
        
        elif args.mode == 'nq':
            upload_nq_data(s3_client, bucket, data_dir, args.overwrite)
        
        elif args.mode == 'simple_qa':
            upload_simple_qa_data(s3_client, bucket, data_dir, args.overwrite)
        
        elif args.mode == 'custom':
            if not args.local_file or not args.s3_key:
                logger.error("Для режима custom необходимо указать --local-file и --s3-key")
                return 1
            
            local_file = Path(args.local_file)
            upload_custom_file(s3_client, bucket, local_file, args.s3_key, args.overwrite)
        
        logger.info("🎉 Загрузка завершена!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
