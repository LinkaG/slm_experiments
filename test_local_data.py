#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест загрузки локальных данных NQ.
"""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.local_nq_dataset import LocalNaturalQuestionsDataset

def test_data_loading():
    """Тестирует загрузку локальных данных NQ."""
    
    # Конфигурация для локального датасета
    config = {
        'name': 'local_natural_questions',
        'type': 'nq',
        'use_local': True,
        'train_path': 'data/nq/nq_full_dataset.json',
        'eval_path': 'data/nq/nq_full_dataset.json',
        'cache_dir': '.cache/datasets'
    }
    
    print("Загружаем локальный датасет NQ...")
    
    try:
        # Создаем датасет
        dataset = LocalNaturalQuestionsDataset(config)
        
        # Получаем статистику
        stats = dataset.get_dataset_stats()
        print(f"Статистика датасета: {stats}")
        
        # Тестируем загрузку eval данных
        print("\nТестируем загрузку eval данных...")
        eval_items = list(dataset.get_eval_data())
        print(f"Загружено {len(eval_items)} eval элементов")
        
        if eval_items:
            # Показываем первый элемент
            first_item = eval_items[0]
            print(f"\nПервый элемент:")
            print(f"  Question: {first_item.question}")
            print(f"  Answer: {first_item.answer}")
            print(f"  Context length: {len(first_item.context) if first_item.context else 0} chars")
            print(f"  Long contexts: {len(first_item.metadata.get('long_context', []))}")
            print(f"  Short contexts: {len(first_item.metadata.get('short_context', []))}")
            print(f"  ID: {first_item.metadata.get('id')}")
            print(f"  URL: {first_item.metadata.get('url')}")
            print(f"  Title: {first_item.metadata.get('title')}")
            
            # Показываем примеры контекстов
            if first_item.metadata.get('long_context'):
                print(f"\nПример long_context:")
                print(f"  {first_item.metadata['long_context'][0][:200]}...")
            
            if first_item.metadata.get('short_context'):
                print(f"\nПримеры short_context:")
                for i, ctx in enumerate(first_item.metadata['short_context'][:3]):
                    print(f"  {i+1}: {ctx}")
        
        print("\n✅ Тест загрузки данных прошел успешно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_data_loading()
    sys.exit(0 if success else 1)
