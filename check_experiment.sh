#!/bin/bash
# Скрипт для проверки статуса фонового эксперимента
# Использование: ./check_experiment.sh [PID_FILE]

LOG_DIR="./logs"

if [ -z "$1" ]; then
    # Ищем последний PID файл
    PID_FILE=$(ls -t ${LOG_DIR}/*.pid 2>/dev/null | head -1)
    if [ -z "$PID_FILE" ]; then
        echo "❌ PID файл не найден в ${LOG_DIR}"
        echo "Использование: $0 [PID_FILE]"
        exit 1
    fi
else
    PID_FILE="$1"
fi

if [ ! -f "$PID_FILE" ]; then
    echo "❌ PID файл не найден: $PID_FILE"
    exit 1
fi

PID=$(cat "$PID_FILE")
LOG_FILE="${PID_FILE%.pid}.log"

if [ -z "$PID" ]; then
    echo "❌ PID файл пуст: $PID_FILE"
    exit 1
fi

echo "📊 Статус эксперимента"
echo "======================"
echo ""

# Проверяем, существует ли процесс
if ps -p "$PID" > /dev/null 2>&1; then
    echo "✅ Процесс запущен (PID: $PID)"
    echo ""
    
    # Показываем информацию о процессе
    ps -p "$PID" -o pid,ppid,cmd,%mem,%cpu,etime
    echo ""
    
    # Показываем размер лог файла
    if [ -f "$LOG_FILE" ]; then
        LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
        echo "📁 Лог файл: $LOG_FILE"
        echo "   Размер: $LOG_SIZE"
        echo ""
        echo "📝 Последние 10 строк лога:"
        echo "---"
        tail -n 10 "$LOG_FILE"
        echo "---"
    else
        echo "⚠️  Лог файл не найден: $LOG_FILE"
    fi
else
    echo "❌ Процесс не найден (PID: $PID)"
    echo "   Возможно, эксперимент уже завершен"
    echo ""
    
    if [ -f "$LOG_FILE" ]; then
        echo "📁 Лог файл: $LOG_FILE"
        echo "📝 Последние 20 строк лога:"
        echo "---"
        tail -n 20 "$LOG_FILE"
        echo "---"
    fi
    
    # Предлагаем удалить PID файл
    read -p "Удалить PID файл? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$PID_FILE"
        echo "✅ PID файл удален"
    fi
fi

