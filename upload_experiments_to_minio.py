#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для загрузки результатов экспериментов из локальных папок в MinIO.

Использование:
    # Загрузить outputs/ -> no-context, output_3/ -> oracle-long-context (по умолчанию)
    poetry run python upload_experiments_to_minio.py

    # Указать свои папки и бакеты
    poetry run python upload_experiments_to_minio.py --source outputs --bucket no-context
    poetry run python upload_experiments_to_minio.py --source output_3 --bucket oracle-long-context

    # Загрузить все предустановленные маппинги (outputs->no-context, output_3->oracle-long-context)
    poetry run python upload_experiments_to_minio.py --all

    # Dry-run (показать что будет загружено, без загрузки)
    poetry run python upload_experiments_to_minio.py --all --dry-run
"""

import argparse
import logging
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Файлы для загрузки в каждом эксперименте
EXPERIMENT_FILES = ["results.json", "predictions.json", "memory_usage.json"]

# Предустановленные маппинги: папка -> bucket (experiment_mode с _ -> -)
DEFAULT_MAPPINGS = [
    ("outputs", "no-context"),
    ("output_3", "oracle-long-context"),
]


def get_s3_client():
    """Создаёт S3 клиент для MinIO."""
    load_dotenv()
    endpoint = os.getenv("CLEARML_S3_ENDPOINT", "http://localhost:9000")
    access_key = os.getenv("CLEARML_S3_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("CLEARML_S3_SECRET_KEY", "minioadmin")
    region = os.getenv("CLEARML_S3_REGION", "us-east-1")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )


def ensure_bucket_exists(s3_client, bucket: str) -> bool:
    """Проверяет существование bucket, создаёт при необходимости."""
    try:
        s3_client.head_bucket(Bucket=bucket)
        logger.info(f"✅ Bucket '{bucket}' существует")
        return True
    except Exception as e:
        error_str = str(e).lower()
        if "404" in error_str or "not found" in error_str or "nosuchbucket" in error_str:
            try:
                s3_client.create_bucket(Bucket=bucket)
                logger.info(f"✅ Bucket '{bucket}' создан")
                return True
            except Exception as create_err:
                logger.error(f"❌ Не удалось создать bucket '{bucket}': {create_err}")
                return False
        else:
            logger.error(f"❌ Ошибка доступа к bucket '{bucket}': {e}")
            return False


def upload_experiment(
    s3_client,
    source_dir: Path,
    bucket: str,
    experiment_name: str,
    dry_run: bool = False,
) -> int:
    """
    Загружает один эксперимент в MinIO.
    Возвращает количество загруженных файлов.
    """
    experiment_dir = source_dir / experiment_name
    if not experiment_dir.is_dir():
        return 0

    uploaded = 0
    for filename in EXPERIMENT_FILES:
        local_file = experiment_dir / filename
        if not local_file.exists():
            logger.debug(f"   Пропуск (нет файла): {local_file}")
            continue

        s3_key = f"{experiment_name}/experiment_results/{filename}"
        if dry_run:
            logger.info(f"   [dry-run] {local_file} -> s3://{bucket}/{s3_key}")
            uploaded += 1
            continue

        try:
            s3_client.upload_file(str(local_file), bucket, s3_key)
            logger.info(f"   ✅ {filename} -> s3://{bucket}/{s3_key}")
            uploaded += 1
        except Exception as e:
            logger.error(f"   ❌ Ошибка загрузки {filename}: {e}")

    return uploaded


def upload_from_source(
    s3_client,
    source_dir: Path,
    bucket: str,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Загружает все эксперименты из source_dir в bucket.
    Возвращает (количество экспериментов, количество файлов).
    """
    if not source_dir.exists():
        logger.warning(f"⚠️ Папка не найдена: {source_dir}")
        return 0, 0

    # Собираем подпапки (каждая = эксперимент)
    experiment_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    if not experiment_dirs:
        logger.warning(f"⚠️ Нет подпапок в {source_dir}")
        return 0, 0

    logger.info(f"📁 {source_dir} -> bucket '{bucket}' ({len(experiment_dirs)} экспериментов)")
    total_files = 0
    for exp_dir in sorted(experiment_dirs):
        n = upload_experiment(s3_client, source_dir, bucket, exp_dir.name, dry_run)
        if n > 0:
            total_files += n

    return len(experiment_dirs), total_files


def main():
    parser = argparse.ArgumentParser(description="Загрузка результатов экспериментов в MinIO")
    parser.add_argument(
        "--source",
        help="Папка с экспериментами (например outputs или output_3)",
    )
    parser.add_argument(
        "--bucket",
        help="Имя bucket в MinIO (например no-context или oracle-long-context)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Загрузить все предустановленные маппинги (outputs->no-context, output_3->oracle-long-context)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Показать что будет загружено, без фактической загрузки",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Корневая папка проекта (по умолчанию текущая)",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()

    if args.all:
        mappings = DEFAULT_MAPPINGS
    elif args.source and args.bucket:
        mappings = [(args.source, args.bucket)]
    else:
        parser.error("Укажите --all или оба --source и --bucket")

    logger.info("🔧 Подключение к MinIO...")
    s3_client = get_s3_client()

    try:
        s3_client.list_buckets()
        logger.info("✅ Подключение к MinIO успешно")
    except Exception as e:
        logger.error(f"❌ Не удалось подключиться к MinIO: {e}")
        logger.error("   Проверьте CLEARML_S3_ENDPOINT, CLEARML_S3_ACCESS_KEY, CLEARML_S3_SECRET_KEY")
        return 1

    total_experiments = 0
    total_files = 0

    for source_name, bucket in mappings:
        source_dir = workspace / source_name
        if not args.dry_run:
            if not ensure_bucket_exists(s3_client, bucket):
                continue
        n_exp, n_files = upload_from_source(s3_client, source_dir, bucket, args.dry_run)
        total_experiments += n_exp
        total_files += n_files

    logger.info(f"📊 Итого: {total_experiments} экспериментов, {total_files} файлов")
    if args.dry_run:
        logger.info("💡 Запустите без --dry-run для фактической загрузки")
    return 0


if __name__ == "__main__":
    exit(main())
