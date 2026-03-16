#!/bin/bash
# Быстрая проверка доступности сервисов через Docker сеть (без установки зависимостей)

set -e

CLEARML_NETWORK="clearml_backend"

if ! docker network inspect "$CLEARML_NETWORK" > /dev/null 2>&1; then
    echo "❌ Docker сеть $CLEARML_NETWORK не найдена"
    exit 1
fi

echo "🔍 Проверка доступности сервисов через Docker сеть $CLEARML_NETWORK"
echo ""

# Проверка ClearML API
echo "1️⃣  Проверка ClearML API (clearml-apiserver:8008)..."
if docker run --rm --network "$CLEARML_NETWORK" curlimages/curl:latest \
    curl -s --connect-timeout 5 http://clearml-apiserver:8008/auth.login 2>&1 | grep -q "result_code"; then
    echo "   ✅ ClearML API доступен"
else
    echo "   ❌ ClearML API недоступен"
fi

# Проверка ClearML Web
echo ""
echo "2️⃣  Проверка ClearML Web (clearml-webserver:80)..."
if docker run --rm --network "$CLEARML_NETWORK" curlimages/curl:latest \
    curl -s --connect-timeout 5 http://clearml-webserver:80 2>&1 | grep -q "ClearML"; then
    echo "   ✅ ClearML Web доступен"
else
    echo "   ❌ ClearML Web недоступен"
fi

# Проверка MinIO
echo ""
echo "3️⃣  Проверка MinIO S3 (minio:9000)..."
if docker run --rm --network "$CLEARML_NETWORK" curlimages/curl:latest \
    curl -s --connect-timeout 5 http://minio:9000 2>&1 | grep -q -E "(MinIO|AccessDenied|Error)"; then
    echo "   ✅ MinIO доступен"
else
    echo "   ❌ MinIO недоступен"
fi

echo ""
echo "✅ Проверка завершена!"
echo ""
echo "💡 Для полного теста с созданием Task используйте:"
echo "   ./run_in_docker_network.sh test_clearml_connection.py"
echo "   или"
echo "   poetry run python test_clearml_connection.py"

