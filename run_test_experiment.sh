#!/bin/bash
# Скрипт для запуска тестового эксперимента на 10 сэмплах с ClearML

echo "🚀 Запуск тестового эксперимента на 10 сэмплах"
echo "================================================"
echo ""
echo "✅ ClearML логирование включено"
echo "📊 Датасет: Natural Questions (NQ)"
echo "🤖 Модель: smollm2_135m (dummy для теста)"
echo "📈 Количество примеров: 10"
echo ""

# Запускаем эксперимент с ClearML
CLEARML_CONFIG_FILE=./clearml.conf poetry run python run_experiment_simple.py

echo ""
echo "================================================"
echo "✅ Эксперимент завершен!"
echo ""
echo "📁 Локальные результаты:"
echo "   outputs/smollm2_135m_rag_nq_full_no_context/"
echo ""
echo "🌐 Результаты в ClearML:"
echo "   http://51.250.43.3:8080"
echo ""

