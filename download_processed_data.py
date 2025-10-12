#!/usr/bin/env python3
"""
Скрипт для скачивания готовых обработанных данных из S3.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_s3_config():
    """Загружает конфигурацию S3 из переменных окружения."""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    config = {
        'endpoint_url': os.getenv('CLEARML_S3_ENDPOINT'),
        'aws_access_key_id': os.getenv('CLEARML_S3_ACCESS_KEY'),
        'aws_secret_access_key': os.getenv('CLEARML_S3_SECRET_KEY'),
        'region_name': os.getenv('CLEARML_S3_REGION', 'us-east-1'),
        'bucket': os.getenv('CLEARML_S3_BUCKET', 'clearml-artifacts'),
        'path_style': os.getenv('CLEARML_S3_PATH_STYLE', 'true').lower() == 'true',
        'verify_ssl': os.getenv('CLEARML_S3_VERIFY_SSL', 'false').lower() == 'true'
    }
    
    return config

def create_s3_client(config):
    """Создает S3 клиент."""
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=config['endpoint_url'],
            aws_access_key_id=config['aws_access_key_id'],
            aws_secret_access_key=config['aws_secret_access_key'],
            region_name=config['region_name'],
            use_ssl=config['verify_ssl']
        )
        return s3_client
    except Exception as e:
        logger.error(f"❌ Ошибка создания S3 клиента: {e}")
        return None

def test_s3_connection(s3_client, bucket):
    """Тестирует подключение к S3."""
    try:
        s3_client.head_bucket(Bucket=bucket)
        logger.info("✅ Подключение к S3 успешно!")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            logger.error(f"❌ Bucket {bucket} не найден")
        else:
            logger.error(f"❌ Ошибка подключения к S3: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к S3: {e}")
        return False

def download_file_from_s3(s3_client, bucket, s3_key, local_file, overwrite=False):
    """Скачивает файл из S3."""
    try:
        # Проверяем, существует ли файл
        if not overwrite and local_file.exists():
            logger.info(f"⏭️  Файл уже существует: {local_file}")
            return True
        
        # Создаем директорию если нужно
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Скачиваем файл
        logger.info(f"📥 Скачиваем s3://{bucket}/{s3_key} -> {local_file}")
        s3_client.download_file(bucket, s3_key, str(local_file))
        logger.info(f"✅ Успешно скачан: {local_file}")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            logger.error(f"❌ Файл не найден в S3: s3://{bucket}/{s3_key}")
        else:
            logger.error(f"❌ Ошибка скачивания s3://{bucket}/{s3_key}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка скачивания {local_file}: {e}")
        return False

def download_processed_nq_data(s3_client, bucket, data_dir, overwrite=False):
    """Скачивает готовые обработанные NQ данные."""
    logger.info("📊 Скачиваем готовые NQ данные...")
    
    # Определяем файлы для скачивания (только доступные в S3)
    nq_files = [
        ('nq_full_dataset.json', 'nq/nq_full_dataset.json')
    ]
    
    success_count = 0
    for s3_key, local_path in nq_files:
        local_file = data_dir / local_path
        if download_file_from_s3(s3_client, bucket, s3_key, local_file, overwrite):
            success_count += 1
    
    logger.info(f"📈 Скачано {success_count} NQ файлов")
    return success_count > 0

def download_processed_simple_qa_data(s3_client, bucket, data_dir, overwrite=False):
    """Скачивает готовые обработанные SimpleQA данные."""
    logger.info("📊 Скачиваем готовые SimpleQA данные...")
    
    # Определяем файлы для скачивания (только доступные в S3)
    simple_qa_files = [
        ('simple_qa_converted.json', 'simple_qa/simple_qa_converted.json')
    ]
    
    success_count = 0
    for s3_key, local_path in simple_qa_files:
        local_file = data_dir / local_path
        if download_file_from_s3(s3_client, bucket, s3_key, local_file, overwrite):
            success_count += 1
    
    logger.info(f"📈 Скачано {success_count} SimpleQA файлов")
    return success_count > 0

def download_processed_both_data(s3_client, bucket, data_dir, overwrite=False):
    """Скачивает готовые обработанные данные для обоих датасетов."""
    logger.info("📊 Скачиваем готовые данные для обоих датасетов...")
    
    nq_success = download_processed_nq_data(s3_client, bucket, data_dir, overwrite)
    simple_qa_success = download_processed_simple_qa_data(s3_client, bucket, data_dir, overwrite)
    
    return nq_success and simple_qa_success

def main():
    parser = argparse.ArgumentParser(description='Скачивание готовых обработанных данных из S3')
    parser.add_argument('--mode', choices=['nq', 'simple_qa', 'both'], default='both',
                       help='Режим скачивания (по умолчанию: both)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Локальная директория для сохранения данных (по умолчанию: data)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Перезаписать существующие файлы')
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
        if not s3_client:
            return 1
        
        bucket = config['bucket']
        
        # Тестируем подключение
        logger.info(f"🔍 Тестируем подключение к S3 bucket: {bucket}")
        if not test_s3_connection(s3_client, bucket):
            return 1
        
        # Создаем директорию для данных
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Скачиваем данные в зависимости от режима
        success = False
        if args.mode == 'nq':
            success = download_processed_nq_data(s3_client, bucket, data_dir, args.overwrite)
        elif args.mode == 'simple_qa':
            success = download_processed_simple_qa_data(s3_client, bucket, data_dir, args.overwrite)
        elif args.mode == 'both':
            success = download_processed_both_data(s3_client, bucket, data_dir, args.overwrite)
        
        if success:
            logger.info("🎉 Скачивание готовых данных завершено успешно!")
            return 0
        else:
            logger.error("❌ Ошибка при скачивании данных")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⏹️  Операция прервана пользователем")
        return 1
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
