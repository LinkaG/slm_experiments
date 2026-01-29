#!/bin/bash
# Быстрый запуск экспериментов через предсобранный Docker образ
# Зависимости уже установлены в образе, поэтому запуск намного быстрее
# Модели кешируются между запусками для ускорения загрузки

set -e

SCRIPT_NAME="${1}"
if [ -z "$SCRIPT_NAME" ]; then
    echo "Использование: $0 <script.py> [args...]"
    echo ""
    echo "💡 Сначала соберите образ:"
    echo "   ./build_docker_image.sh"
    echo ""
    echo "💾 Кеш моделей хранится в: ~/.cache/docker-models/"
    echo "   Модели HuggingFace будут кешироваться между запусками"
    exit 1
fi
shift

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "❌ Файл $SCRIPT_NAME не найден"
    exit 1
fi

CLEARML_NETWORK="clearml_backend"
IMAGE_NAME="slm-experiments:latest"

# Определяем, использовать ли GPU
# Проверяем доступность nvidia-container-toolkit
USE_GPU=false
if docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    USE_GPU=true
    echo "🎯 GPU доступен, будет использоваться для ускорения"
else
    echo "💻 GPU недоступен в Docker, будет использоваться CPU"
fi

# Проверяем, существует ли образ
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "❌ Docker образ $IMAGE_NAME не найден"
    echo ""
    echo "💡 Соберите образ командой:"
    echo "   ./build_docker_image.sh"
    exit 1
fi

if ! docker network inspect "$CLEARML_NETWORK" > /dev/null 2>&1; then
    echo "❌ Docker сеть $CLEARML_NETWORK не найдена"
    exit 1
fi

echo "🚀 Быстрый запуск $SCRIPT_NAME через Docker сеть $CLEARML_NETWORK"
echo "📦 Используется предсобранный образ (зависимости уже установлены)"
echo "⚡ Запуск будет намного быстрее!"

# Создаем директорию для кеша моделей на хосте (если не существует)
# Можно переопределить через переменную окружения DOCKER_MODELS_CACHE
CACHE_DIR="${DOCKER_MODELS_CACHE:-${HOME}/.cache/docker-models}"
mkdir -p "$CACHE_DIR/huggingface"
mkdir -p "$CACHE_DIR/datasets"
echo "💾 Кеш моделей: $CACHE_DIR"
echo "   Модели HuggingFace будут сохраняться между запусками"

# Собираем все аргументы
ARGS="$@"

# Запускаем в предсобранном образе
# Монтируем конфигурацию напрямую в ~/.clearml.conf
# Добавляем поддержку GPU если доступна
# Монтируем кеш моделей для ускорения загрузки
DOCKER_ARGS="--rm --network $CLEARML_NETWORK"
if [ "$USE_GPU" = true ]; then
    DOCKER_ARGS="$DOCKER_ARGS --gpus all"
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
    "$IMAGE_NAME" \
    bash -c "
        echo '✅ Конфигурация ClearML смонтирована в ~/.clearml.conf'
        echo '✅ Переменные окружения для MinIO установлены'
        echo '✅ Кеш моделей смонтирован: /root/.cache/huggingface'
        echo '💾 Модели будут кешироваться между запусками'
        python $SCRIPT_NAME $ARGS
    "

