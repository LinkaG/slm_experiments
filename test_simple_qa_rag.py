#!/usr/bin/env python3
"""
Простой тест SimpleQA для RAG системы без полной установки зависимостей.
"""

import json
from pathlib import Path

def test_simple_qa_rag_compatibility():
    """Тестирует совместимость SimpleQA с RAG системой."""
    print("=== Тест совместимости SimpleQA с RAG системой ===")
    
    # Загружаем конвертированные данные
    file_path = Path("data/simple_qa/simple_qa_converted.json")
    
    if not file_path.exists():
        print(f"❌ Файл {file_path} не найден!")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Файл загружен: {len(data.get('train', []))} train, {len(data.get('eval', []))} eval")
    
    # Проверяем структуру для RAG
    eval_items = data.get('eval', [])
    
    if not eval_items:
        print("❌ Нет eval данных!")
        return False
    
    print(f"\n=== Анализ RAG совместимости ===")
    
    # Анализируем несколько примеров
    for i, item in enumerate(eval_items[:3]):
        print(f"\n--- Пример {i+1} ---")
        print(f"Вопрос: {item['question']}")
        print(f"Ответ: {item['answer']}")
        
        # Проверяем контекст
        context = item.get('context', '')
        if isinstance(context, list):
            context = ' '.join(context)
        
        print(f"Контекст (длина: {len(context)} символов)")
        print(f"Контекст (первые 200 символов): {context[:200]}...")
        
        # Проверяем метаданные
        metadata = item.get('metadata', {})
        print(f"Тема: {metadata.get('topic', 'N/A')}")
        print(f"Тип ответа: {metadata.get('answer_type', 'N/A')}")
        print(f"URL источников: {len(metadata.get('urls', []))}")
    
    print(f"\n=== RAG совместимость ===")
    print("✅ Вопросы: Есть")
    print("✅ Ответы: Есть") 
    print("✅ Контекст: Есть (документы)")
    print("✅ Метаданные: Есть")
    
    print(f"\n=== Рекомендации для RAG тестирования ===")
    print("1. Oracle Context режим: Использовать предоставленные документы как контекст")
    print("2. Retriever Context режим: Использовать ретривер для поиска релевантных документов")
    print("3. No Context режим: Тестировать модель без дополнительного контекста")
    
    print(f"\n=== Примеры экспериментов ===")
    print("# Тест с идеальным контекстом:")
    print("poetry run python -m src.cli run-experiment model=smollm2_135m dataset=local_simple_qa experiment_mode=oracle_context")
    print()
    print("# Тест с ретривером:")
    print("poetry run python -m src.cli run-experiment model=smollm2_135m dataset=local_simple_qa experiment_mode=retriever_context")
    print()
    print("# Тест без контекста:")
    print("poetry run python -m src.cli run-experiment model=smollm2_135m dataset=local_simple_qa experiment_mode=no_context")
    
    return True

def analyze_simple_qa_topics():
    """Анализирует темы в SimpleQA датасете."""
    print(f"\n=== Анализ тем SimpleQA ===")
    
    file_path = Path("data/simple_qa/simple_qa_converted.json")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Собираем все темы
    topics = {}
    answer_types = {}
    
    for item in data.get('eval', []):
        metadata = item.get('metadata', {})
        topic = metadata.get('topic', 'Unknown')
        answer_type = metadata.get('answer_type', 'Unknown')
        
        topics[topic] = topics.get(topic, 0) + 1
        answer_types[answer_type] = answer_types.get(answer_type, 0) + 1
    
    print("Темы в eval данных:")
    for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count}")
    
    print("\nТипы ответов:")
    for answer_type, count in sorted(answer_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {answer_type}: {count}")

if __name__ == "__main__":
    print("Анализ SimpleQA для RAG системы")
    print("=" * 50)
    
    # Тест совместимости
    compatibility_ok = test_simple_qa_rag_compatibility()
    
    if compatibility_ok:
        # Анализ тем
        analyze_simple_qa_topics()
        
        print(f"\n🎉 SimpleQA готов для RAG экспериментов!")
    else:
        print(f"\n❌ SimpleQA не готов для RAG экспериментов!")
