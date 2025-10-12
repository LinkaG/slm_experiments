#!/usr/bin/env python3
"""
Скрипт для проверки артефактов в MinIO bucket
"""

import boto3
from botocore.client import Config
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Настройки MinIO
MINIO_ENDPOINT = os.getenv('CLEARML_S3_ENDPOINT', 'http://51.250.43.3:9000')
MINIO_ACCESS_KEY = os.getenv('CLEARML_S3_ACCESS_KEY', 'minio_admin_2024')
MINIO_SECRET_KEY = os.getenv('CLEARML_S3_SECRET_KEY', 'Kx9mP7$vL2@nQ8!wE5&rT3*yU6+iO1-pA4^sD9~fG0')
BUCKET_NAME = 'clearml-artifacts'

print(f"🔍 Проверка содержимого MinIO bucket: {BUCKET_NAME}")
print(f"📍 Endpoint: {MINIO_ENDPOINT}")
print("=" * 60)

try:
    # Создаем S3 клиент для MinIO
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1',
        verify=False
    )
    
    # Проверяем существование bucket
    print(f"\n📦 Checking bucket '{BUCKET_NAME}'...")
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print(f"✅ Bucket '{BUCKET_NAME}' exists")
    except Exception as e:
        print(f"❌ Bucket '{BUCKET_NAME}' not found or not accessible")
        print(f"Error: {e}")
        exit(1)
    
    # Список объектов в bucket
    print(f"\n📂 Contents of '{BUCKET_NAME}':")
    print("-" * 60)
    
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME)
    
    if 'Contents' in response:
        total_size = 0
        file_count = 0
        
        for obj in response['Contents']:
            size_mb = obj['Size'] / (1024 * 1024)
            total_size += obj['Size']
            file_count += 1
            
            print(f"\n📄 {obj['Key']}")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Modified: {obj['LastModified']}")
        
        print("\n" + "=" * 60)
        print(f"📊 Summary:")
        print(f"   Total files: {file_count}")
        print(f"   Total size: {total_size / (1024 * 1024):.2f} MB")
        
        # Проверяем наличие артефактов эксперимента
        artifact_files = [obj['Key'] for obj in response['Contents'] if 'json' in obj['Key'].lower()]
        if artifact_files:
            print(f"\n✅ Found {len(artifact_files)} artifact file(s):")
            for f in artifact_files:
                print(f"   - {f}")
        else:
            print("\n⚠️  No JSON artifacts found")
            
    else:
        print("📭 Bucket is empty")
    
    print("\n" + "=" * 60)
    print("✅ Check completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

