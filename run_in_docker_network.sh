#!/bin/bash
# Упрощенный скрипт для запуска через Docker сеть
# Использует существующий контейнер или создает минимальный

set -e

SCRIPT_NAME="${1}"
if [ -z "$SCRIPT_NAME" ]; then
    echo "Использование: $0 <script.py> [args...]"
    exit 1
fi
shift

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "❌ Файл $SCRIPT_NAME не найден"
    exit 1
fi

CLEARML_NETWORK="clearml_backend"

if ! docker network inspect "$CLEARML_NETWORK" > /dev/null 2>&1; then
    echo "❌ Docker сеть $CLEARML_NETWORK не найдена"
    exit 1
fi

# Определяем, использовать ли GPU
USE_GPU=false
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    USE_GPU=true
    echo "🎯 GPU доступен, будет использоваться для ускорения"
else
    echo "💻 GPU недоступен в Docker, будет использоваться CPU"
fi

echo "🚀 Запуск $SCRIPT_NAME через Docker сеть $CLEARML_NETWORK"
echo "📦 Используется конфигурация: clearml.conf.docker"

# Создаем директорию для кеша моделей на хосте (если не существует)
# Можно переопределить через переменную окружения DOCKER_MODELS_CACHE
CACHE_DIR="${DOCKER_MODELS_CACHE:-/storage/docker-models}"
mkdir -p "$CACHE_DIR/huggingface"
mkdir -p "$CACHE_DIR/datasets"
echo "💾 Кеш моделей: $CACHE_DIR"

# Собираем все аргументы в одну строку, правильно экранируя
ARGS="$@"

# Запускаем в временном контейнере
# Монтируем конфигурацию напрямую в ~/.clearml.conf
# Добавляем поддержку GPU если доступна
# Монтируем кеш моделей для ускорения загрузки
DOCKER_ARGS="--rm --network $CLEARML_NETWORK"
if [ "$USE_GPU" = true ]; then
    DOCKER_ARGS="$DOCKER_ARGS --gpus all"
    # Используем образ с CUDA для GPU
    BASE_IMAGE="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
else
    # Используем легкий образ для CPU
    BASE_IMAGE="python:3.10-slim"
fi

docker run $DOCKER_ARGS \
    -v "$(pwd):/workspace" \
    -v "$(pwd)/clearml.conf.docker:/root/.clearml.conf:ro" \
    -v "$(pwd)/.env:/workspace/.env:ro" \
    -v "$CACHE_DIR/huggingface:/root/.cache/huggingface" \
    -v "$CACHE_DIR/datasets:/root/.cache/datasets" \
    -w /workspace \
    -e PYTHONPATH=/workspace \
    -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e CLEARML_S3_ENDPOINT=http://minio:9000 \
    -e CLEARML_S3_BUCKET=clearml-artifacts \
    -e CLEARML_S3_ACCESS_KEY=minioadmin \
    -e CLEARML_S3_SECRET_KEY=minioadmin \
    -e CLEARML_S3_REGION=us-east-1 \
    "$BASE_IMAGE" \
    bash -c "
        if [ \"$USE_GPU\" = false ]; then
            pip install -q --no-cache-dir clearml boto3 python-dotenv requests omegaconf hydra-core torch transformers tqdm pandas
        else
            pip install -q --no-cache-dir clearml boto3 python-dotenv requests omegaconf hydra-core transformers tqdm pandas
        fi
        echo '✅ Конфигурация ClearML смонтирована в ~/.clearml.conf'
        echo '✅ Переменные окружения для MinIO установлены'
        echo '✅ Кеш моделей смонтирован: /root/.cache/huggingface'
        echo '💾 Модели будут кешироваться между запусками'
        python $SCRIPT_NAME $ARGS
    "

