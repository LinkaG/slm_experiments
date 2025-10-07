#!/bin/bash
# Скрипт для настройки переменных окружения ClearML

export CLEARML_API_HOST="http://51.250.43.3:8008"
export CLEARML_WEB_HOST="http://51.250.43.3:8080"
export CLEARML_FILES_HOST="http://51.250.43.3:8081"
export CLEARML_API_ACCESS_KEY="UTO3G5160MC40B3IB5WVAZUJ70PBKH"
export CLEARML_API_SECRET_KEY="zoPVNzfWQWjx5ae-g5TuiZzUPDucgVv8xFBPEYBJRJA2C5A498glzwASDrtGxV-0QlI"

# S3 настройки
export CLEARML_S3_ENDPOINT="http://51.250.43.3:9000"
export CLEARML_S3_BUCKET="clearml-artifacts"
export CLEARML_S3_ACCESS_KEY="minio_admin_2024"
export CLEARML_S3_SECRET_KEY="Kx9mP7\$vL2@nQ8!wE5&rT3*yU6+iO1-pA4^sD9~fG0"

echo "✅ Переменные окружения ClearML настроены!"
echo "🔗 Веб-интерфейс: $CLEARML_WEB_HOST"
echo "🔗 API: $CLEARML_API_HOST"
echo "🔗 Файлы: $CLEARML_FILES_HOST"
