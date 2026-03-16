#!/usr/bin/env python3
"""
Тест загрузки локальных данных SimpleQA.
"""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / 'src'))

from data.local_simple_qa_dataset import LocalSimpleQADataset

def test_data_loading():
    """Тестирует загрузку локальных данных SimpleQA."""
    
    # Конфигурация для локального датасета
    config = {
        'name': 'local_simple_qa',
        'type': 'simple_qa',
        'use_local': True,
        'train_path': 'data/simple_qa/simple_qa_converted.json',
        'eval_path': 'data/simple_qa/simple_qa_converted.json',
        'cache_dir': '.cache/datasets'
    }
    
    print("Загружаем локальный датасет SimpleQA...")
    
    try:
        # Создаем датасет
        dataset = LocalSimpleQADataset(config)
        
        # Выводим статистику
        print(f"Статистика датасета: {dataset.get_dataset_stats()}")
        
        # Тестируем загрузку eval данных
        print("\nТестируем загрузку eval данных...")
        eval_data = list(dataset.get_eval_data())
        print(f"Загружено {len(eval_data)} eval элементов")
        
        # Показываем первый элемент
        if eval_data:
            first_item = eval_data[0]
            print(f"\nПервый элемент:")
            print(f"  Question: {first_item.question}")
            print(f"  Answer: {first_item.answer}")
            print(f"  Context length: {len(first_item.context) if first_item.context else 0} chars")
            print(f"  ID: {first_item.metadata.get('id', 'N/A')}")
            
            # Показываем примеры контекста
            context = first_item.context
            if context and len(context) > 200:
                print(f"\nПример context (первые 200 символов):")
                print(f"  {context[:200]}...")
            elif context:
                print(f"\nContext:")
                print(f"  {context}")
            else:
                print(f"\nContext: None")
        
        print("\n✅ Тест загрузки данных прошел успешно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        return False

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
