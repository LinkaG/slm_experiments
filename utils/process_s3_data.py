#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обработки данных из S3.

Скачивает исходные файлы из S3, конвертирует их в нужный формат
и загружает обратно в S3.

Использование:
    poetry run python process_s3_data.py --help
"""

import boto3
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import logging
from dotenv import load_dotenv
import os
import tempfile
import subprocess
import sys

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

def download_from_s3(s3_client: boto3.client, bucket: str, s3_key: str, local_file: Path) -> bool:
    """Скачивает файл из S3."""
    try:
        logger.info(f"📥 Скачиваем s3://{bucket}/{s3_key} -> {local_file}")
        
        # Создаем директорию если не существует
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        s3_client.download_file(bucket, s3_key, str(local_file))
        logger.info(f"✅ Успешно скачан: {local_file}")
        return True
        
    except s3_client.exceptions.NoSuchKey:
        logger.error(f"❌ Файл не найден в S3: s3://{bucket}/{s3_key}")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка скачивания s3://{bucket}/{s3_key}: {e}")
        return False

def upload_to_s3(s3_client: boto3.client, bucket: str, local_file: Path, s3_key: str) -> bool:
    """Загружает файл в S3."""
    try:
        logger.info(f"📤 Загружаем {local_file} -> s3://{bucket}/{s3_key}")
        
        with open(local_file, 'rb') as f:
            s3_client.upload_fileobj(f, bucket, s3_key)
        
        logger.info(f"✅ Успешно загружен: s3://{bucket}/{s3_key}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки {local_file}: {e}")
        return False

def process_nq_data(s3_client: boto3.client, bucket: str, work_dir: Path, upload_results: bool = True):
    """Обрабатывает NQ данные: скачивает исходный файл, конвертирует, загружает результат."""
    logger.info("🔄 Обрабатываем Natural Questions данные...")
    
    # Определяем целевую папку
    if upload_results:
        # Если загружаем в S3, используем рабочую директорию
        target_dir = work_dir / 'data' / 'nq'
    else:
        # Если не загружаем, используем основную папку data/
        target_dir = Path('data') / 'nq'
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Скачиваем исходный файл
    original_file = target_dir / 'NQ-open.dev.merged.jsonl'
    if not download_from_s3(s3_client, bucket, 'NQ-open.dev.merged.jsonl', original_file):
        logger.error("❌ Не удалось скачать исходный файл NQ")
        return False
    
    # 2. Конвертируем данные
    logger.info("🔄 Конвертируем NQ данные...")
    converted_file = target_dir / 'nq_full_dataset.json'
    
    try:
        # Запускаем скрипт конвертации
        cmd = [
            sys.executable, 'convert_nq_data.py',
            '--input', str(original_file),
            '--output', str(converted_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            logger.error(f"❌ Ошибка конвертации NQ: {result.stderr}")
            return False
        
        logger.info("✅ NQ данные успешно сконвертированы")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при конвертации NQ: {e}")
        return False
    
    # 3. Загружаем результат в S3 (если нужно)
    if upload_results:
        if not upload_to_s3(s3_client, bucket, converted_file, 'nq_full_dataset.json'):
            logger.error("❌ Не удалось загрузить конвертированный файл NQ")
            return False
    
    logger.info("✅ Обработка NQ данных завершена")
    return True

def process_simple_qa_data(s3_client: boto3.client, bucket: str, work_dir: Path, upload_results: bool = True):
    """Обрабатывает SimpleQA данные: скачивает исходный файл, конвертирует, загружает результат."""
    logger.info("🔄 Обрабатываем SimpleQA данные...")
    
    # Определяем целевую папку
    if upload_results:
        # Если загружаем в S3, используем рабочую директорию
        target_dir = work_dir / 'data' / 'simple_qa'
    else:
        # Если не загружаем, используем основную папку data/
        target_dir = Path('data') / 'simple_qa'
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Скачиваем исходный файл
    original_file = target_dir / 'simple_qa_test_set_with_documents.csv'
    if not download_from_s3(s3_client, bucket, 'simple_qa_test_set_with_documents.csv', original_file):
        logger.error("❌ Не удалось скачать исходный файл SimpleQA")
        return False
    
    # 2. Конвертируем данные
    logger.info("🔄 Конвертируем SimpleQA данные...")
    converted_file = target_dir / 'simple_qa_converted.json'
    
    try:
        # Запускаем скрипт конвертации
        cmd = [
            sys.executable, 'convert_simple_qa_data.py',
            '--input', str(original_file),
            '--output', str(converted_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            logger.error(f"❌ Ошибка конвертации SimpleQA: {result.stderr}")
            return False
        
        logger.info("✅ SimpleQA данные успешно сконвертированы")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при конвертации SimpleQA: {e}")
        return False
    
    # 3. Загружаем результат в S3 (если нужно)
    if upload_results:
        if not upload_to_s3(s3_client, bucket, converted_file, 'simple_qa_converted.json'):
            logger.error("❌ Не удалось загрузить конвертированный файл SimpleQA")
            return False
    
    logger.info("✅ Обработка SimpleQA данных завершена")
    return True

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
    parser = argparse.ArgumentParser(description='Обработка данных из S3')
    parser.add_argument('--mode', choices=['nq', 'simple_qa', 'both'], required=True,
                       help='Режим обработки: nq, simple_qa, both')
    parser.add_argument('--work-dir', default='temp_s3_processing',
                       help='Рабочая директория (по умолчанию: temp_s3_processing)')
    parser.add_argument('--no-upload', action='store_true',
                       help='Не загружать результаты обратно в S3')
    parser.add_argument('--bucket',
                       help='S3 bucket (переопределяет .env)')
    parser.add_argument('--cleanup', action='store_true',
                       help='Удалить рабочую директорию после обработки')
    
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
        
        # Создаем рабочую директорию
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Рабочая директория: {work_dir}")
        
        success = True
        
        # Обрабатываем данные
        if args.mode in ['nq', 'both']:
            if not process_nq_data(s3_client, bucket, work_dir, not args.no_upload):
                success = False
        
        if args.mode in ['simple_qa', 'both']:
            if not process_simple_qa_data(s3_client, bucket, work_dir, not args.no_upload):
                success = False
        
        # Очистка рабочей директории
        if args.cleanup:
            logger.info(f"🧹 Удаляем рабочую директорию: {work_dir}")
            import shutil
            shutil.rmtree(work_dir)
        
        if success:
            logger.info("🎉 Обработка данных завершена успешно!")
            return 0
        else:
            logger.error("❌ Обработка данных завершилась с ошибками!")
            return 1
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
