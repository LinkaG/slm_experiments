#!/usr/bin/env python3
"""
Конвертер для SimpleQA датасета в формат, совместимый с RAG фреймворком.
SimpleQA содержит вопросы, ответы и релевантные документы - идеально для RAG тестирования.
"""

import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import ast

def parse_metadata(metadata_str: str) -> Dict[str, Any]:
    """Парсит строку метаданных в словарь."""
    try:
        # Используем ast.literal_eval для безопасного парсинга
        return ast.literal_eval(metadata_str)
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Could not parse metadata '{metadata_str[:50]}...': {e}")
        return {}

def convert_simple_qa_to_rag_format(
    input_csv: str, 
    output_json: str, 
    max_items: int = None
) -> None:
    """
    Конвертирует SimpleQA CSV в формат, совместимый с RAG фреймворком.
    
    Args:
        input_csv: Путь к входному CSV файлу SimpleQA
        output_json: Путь к выходному JSON файлу
        max_items: Максимальное количество элементов для обработки
    """
    
    # Увеличиваем лимит размера поля для больших документов
    csv.field_size_limit(10000000)  # 10MB лимит
    
    items = []
    
    print(f"Читаем SimpleQA данные из {input_csv}...")
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            if max_items and i >= max_items:
                break
                
            # Парсим метаданные
            metadata = parse_metadata(row['metadata'])
            
            # Создаем элемент в формате RAG фреймворка
            item = {
                'question': row['problem'],
                'answer': row['answer'],
                'context': row['documents'],  # Документы как контекст
                'metadata': {
                    'id': i,
                    'topic': metadata.get('topic', ''),
                    'answer_type': metadata.get('answer_type', ''),
                    'urls': metadata.get('urls', []),
                    'source': 'simple_qa'
                }
            }
            
            items.append(item)
            
            if (i + 1) % 100 == 0:
                print(f"Обработано {i + 1} элементов...")
    
    print(f"Всего обработано {len(items)} элементов")
    
    # Сохраняем результат как массив (аналогично NQ)
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    
    print(f"Данные сохранены в {output_path}")
    
    # Выводим статистику
    print("\nСтатистика датасета:")
    print(f"Всего элементов: {len(items)}")
    
    # Примеры данных
    print("\nПримеры вопросов:")
    for i, item in enumerate(items[:3]):
        print(f"{i+1}. {item['question']}")
        print(f"   Ответ: {item['answer']}")
        print(f"   Тема: {item['metadata']['topic']}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Конвертер SimpleQA для RAG фреймворка')
    parser.add_argument('--input', '-i', 
                       default='data/simple_qa/simple_qa_test_set_with_documents.csv',
                       help='Путь к входному CSV файлу')
    parser.add_argument('--output', '-o',
                       default='data/simple_qa/simple_qa_converted.json',
                       help='Путь к выходному JSON файлу')
    parser.add_argument('--max-items', type=int,
                       help='Максимальное количество элементов для обработки')
    
    args = parser.parse_args()
    
    convert_simple_qa_to_rag_format(
        input_csv=args.input,
        output_json=args.output,
        max_items=args.max_items
    )

if __name__ == '__main__':
    main()
