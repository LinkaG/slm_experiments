#!/usr/bin/env python3
"""
Тест загрузки и обработки SimpleQA датасета для RAG фреймворка.
"""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent / "src"))

from data import get_dataset
import json

def test_simple_qa_loading():
    """Тестирует загрузку SimpleQA датасета."""
    print("=== Тест загрузки SimpleQA датасета ===")
    
    # Конфигурация для локального SimpleQA
    dataset_config = {
        'name': 'local_simple_qa',
        'type': 'simple_qa',
        'use_local': True,
        'train_path': 'data/simple_qa/simple_qa_converted.json',
        'eval_path': 'data/simple_qa/simple_qa_converted.json',
        'cache_dir': '.cache/datasets'
    }
    
    try:
        # Создаем датасет
        print("Создаем датасет...")
        dataset = get_dataset(dataset_config)
        
        # Получаем статистику
        print("Получаем статистику...")
        stats = dataset.get_dataset_stats()
        print(f"Статистика датасета: {stats}")
        
        # Тестируем train данные
        print("\n=== Train данные ===")
        train_items = list(dataset.get_train_data())
        print(f"Загружено {len(train_items)} train элементов")
        
        if train_items:
            item = train_items[0]
            print(f"Пример train элемента:")
            print(f"  Вопрос: {item.question}")
            print(f"  Ответ: {item.answer}")
            print(f"  Контекст (первые 100 символов): {item.context[:100] if item.context else 'None'}...")
            print(f"  Метаданные: {item.metadata}")
        
        # Тестируем eval данные
        print("\n=== Eval данные ===")
        eval_items = list(dataset.get_eval_data())
        print(f"Загружено {len(eval_items)} eval элементов")
        
        if eval_items:
            item = eval_items[0]
            print(f"Пример eval элемента:")
            print(f"  Вопрос: {item.question}")
            print(f"  Ответ: {item.answer}")
            print(f"  Контекст (первые 100 символов): {item.context[:100] if item.context else 'None'}...")
            print(f"  Метаданные: {item.metadata}")
        
        print("\n✅ SimpleQA датасет успешно загружен и обработан!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке SimpleQA датасета: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_qa_format():
    """Тестирует формат конвертированного SimpleQA файла."""
    print("\n=== Тест формата SimpleQA файла ===")
    
    file_path = Path("data/simple_qa/simple_qa_converted.json")
    
    if not file_path.exists():
        print(f"❌ Файл {file_path} не найден!")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Структура файла:")
        print(f"  Ключи: {list(data.keys())}")
        
        if 'train' in data:
            print(f"  Train элементов: {len(data['train'])}")
        if 'eval' in data:
            print(f"  Eval элементов: {len(data['eval'])}")
        
        # Проверяем структуру элемента
        if data.get('train'):
            item = data['train'][0]
            print(f"\nСтруктура элемента:")
            print(f"  Ключи: {list(item.keys())}")
            print(f"  Вопрос: {item.get('question', 'N/A')[:50]}...")
            print(f"  Ответ: {item.get('answer', 'N/A')}")
            print(f"  Есть контекст: {bool(item.get('context'))}")
            print(f"  Метаданные: {list(item.get('metadata', {}).keys())}")
        
        print("✅ Формат SimpleQA файла корректен!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при проверке формата: {e}")
        return False

if __name__ == "__main__":
    print("Тестирование SimpleQA датасета для RAG фреймворка")
    print("=" * 50)
    
    # Тест формата файла
    format_ok = test_simple_qa_format()
    
    if format_ok:
        # Тест загрузки датасета
        loading_ok = test_simple_qa_loading()
        
        if loading_ok:
            print("\n🎉 Все тесты SimpleQA прошли успешно!")
        else:
            print("\n❌ Тесты SimpleQA не прошли!")
    else:
        print("\n❌ Формат SimpleQA файла некорректен!")
