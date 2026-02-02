#!/bin/bash
# Скрипт для остановки всех экспериментов, запущенных через run_batch_experiments.py

echo "🔍 Поиск запущенных Docker контейнеров с экспериментами..."

# Находим все контейнеры с образом slm-experiments
CONTAINERS=$(docker ps --filter "ancestor=slm-experiments:latest" --format "{{.ID}}")

if [ -z "$CONTAINERS" ]; then
    echo "✅ Нет запущенных экспериментов"
    exit 0
fi

echo "📋 Найдено контейнеров: $(echo "$CONTAINERS" | wc -l)"
echo ""

# Останавливаем все контейнеры
for CONTAINER in $CONTAINERS; do
    echo "🛑 Остановка контейнера $CONTAINER..."
    docker stop "$CONTAINER"
done

echo ""
echo "✅ Все эксперименты остановлены"

# Показываем текущее состояние GPU
echo ""
echo "📊 Текущее состояние GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s: %s | Память: %s/%s MB | Загрузка: %s%%\n", $1, $2, $3, $4, $5}'
