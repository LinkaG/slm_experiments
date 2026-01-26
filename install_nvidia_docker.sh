#!/bin/bash
# Установка nvidia-container-toolkit для работы GPU в Docker
# Использование: sudo ./install_nvidia_docker.sh

set -e

echo "🔧 Установка nvidia-container-toolkit..."
echo ""

# Проверка прав root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Ошибка: скрипт должен запускаться с правами root (sudo)"
    exit 1
fi

# Определяем дистрибутив
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
echo "📋 Дистрибутив: $distribution"
echo ""

# Добавляем GPG ключ
echo "📝 Добавление GPG ключа..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Добавляем репозиторий
echo "📦 Добавление репозитория..."
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Обновляем список пакетов
echo "🔄 Обновление списка пакетов..."
# Игнорируем ошибки с локальными репозиториями (они не критичны)
apt-get update || true

# Устанавливаем nvidia-container-toolkit
echo "📥 Установка nvidia-container-toolkit..."
apt-get install -y nvidia-container-toolkit

# Настраиваем Docker runtime
echo "⚙️  Настройка Docker runtime..."
nvidia-ctk runtime configure --runtime=docker

# Перезапускаем Docker
echo "🔄 Перезапуск Docker..."
systemctl restart docker

echo ""
echo "✅ Установка завершена!"
echo ""
echo "🧪 Проверка (запустите от обычного пользователя):"
echo "   docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi"
