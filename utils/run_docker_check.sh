#!/bin/bash
# Запуск диагностики окружения в Docker (как при run_batch_experiments)
set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="slm-experiments:latest"

echo "=== Test 1: Без монтирования workspace (чистый образ) ==="
docker run --rm --gpus all "$IMAGE_NAME" python -c "
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
from transformers import AutoModelForCausalLM
print('AutoModelForCausalLM: OK')
"

echo ""
echo "=== Test 2: С монтированием workspace и PYTHONPATH (как в run_batch_experiments) ==="
docker run --rm --gpus all \
  -v "$WORKSPACE_DIR:/workspace" \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  "$IMAGE_NAME" python check_docker_env.py
