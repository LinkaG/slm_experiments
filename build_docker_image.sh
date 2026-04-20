#!/bin/bash
# Скрипт для сборки Docker образа с предустановленными зависимостями

set -e

IMAGE_NAME="slm-experiments"
IMAGE_TAG="latest"

echo "🔨 Сборка Docker образа $IMAGE_NAME:$IMAGE_TAG..."
echo "📦 Используется базовый образ PyTorch с CUDA поддержкой"
echo "💡 Образ будет работать на GPU (если доступен) и CPU (fallback)"
echo "⏱️  Это займет несколько минут при первом запуске"
echo ""

# Используем --no-cache для полной пересборки (важно при изменении версий библиотек)
docker build --no-cache -f Dockerfile.experiments -t "$IMAGE_NAME:$IMAGE_TAG" .

echo ""
echo "✅ Образ собран успешно!"
echo ""
echo "💡 Запуск пакета экспериментов:"
echo "   poetry run python run_batch_experiments.py"
echo "   (или одиночный прогон: docker run ... python run_experiment_simple.py ...)"

