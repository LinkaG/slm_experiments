#!/bin/bash
# Скрипт для настройки переменных окружения ClearML

export CLEARML_API_HOST="http://localhost:8008"
export CLEARML_WEB_HOST="http://localhost:8080"
export CLEARML_FILES_HOST="http://localhost:8081"
export CLEARML_API_ACCESS_KEY="Y5MPEHCJ8FKJ1Z85QH1BMB54YBKVYR"
export CLEARML_API_SECRET_KEY="hXy6h6ROPhY-Pn0jmL7FkNUtak4uhn-RErPFpYYqdYov5ya2pbUlDddCp7O5Lc51R98"

# S3 настройки
export CLEARML_S3_ENDPOINT="http://localhost:9000"
export CLEARML_S3_BUCKET="clearml-artifacts"
export CLEARML_S3_ACCESS_KEY="minioadmin"
export CLEARML_S3_SECRET_KEY="minioadmin"

echo "✅ Переменные окружения ClearML настроены!"
echo "🔗 Веб-интерфейс: $CLEARML_WEB_HOST"
echo "🔗 API: $CLEARML_API_HOST"
echo "🔗 Файлы: $CLEARML_FILES_HOST"
