#!/bin/bash
# Скрипт для запуска recalculate_token_recall.py через Docker сеть

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="recalculate_token_recall.py"

echo "🔄 Запуск пересчета token_recall через Docker сеть"
echo ""

# Запускаем через run_in_docker_network.sh
"$SCRIPT_DIR/run_in_docker_network.sh" "$SCRIPT_NAME" "$@"

