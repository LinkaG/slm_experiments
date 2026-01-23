#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конвертер данных Natural Questions в формат для slm_experiments.

Конвертирует NQ-open.dev.merged.jsonl в формат, совместимый с LocalNaturalQuestionsDataset.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_contexts(item: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Извлекает long_context и short_context из аннотаций.
    
    Args:
        item: Элемент данных NQ
        
    Returns:
        tuple: (long_contexts, short_contexts) - списки извлеченных контекстов
    """
    # Разделяем document_text на токены
    document_text = item.get('document_text', '')
    if not document_text:
        return [], []
    tokens = document_text.split()
    
    long_contexts = []
    short_contexts = []
    
    annotations = item.get('annotations', [])
    if not annotations:
        return [], []
    
    for annotation in annotations:
        # Long context
        if annotation.get('long_answer') is not None:
            long_answer = annotation['long_answer']
            if long_answer.get('candidate_index', -1) != -1:
                cand_idx = long_answer['candidate_index']
                if cand_idx < len(item.get('long_answer_candidates', [])):
                    cand = item['long_answer_candidates'][cand_idx]
                    if cand.get('start_token', -1) != -1 and cand.get('end_token', -1) != -1:
                        if cand['start_token'] < len(tokens) and cand['end_token'] <= len(tokens):
                            text = " ".join(tokens[cand['start_token']:cand['end_token']])
                            if text.strip():  # Добавляем только непустые контексты
                                long_contexts.append(text.strip())
        
        # Short contexts
        short_answers = annotation.get('short_answers', [])
        if short_answers:
            for short_ans in short_answers:
                if short_ans.get('start_token', -1) != -1 and short_ans.get('end_token', -1) != -1:
                    if short_ans['start_token'] < len(tokens) and short_ans['end_token'] <= len(tokens):
                        text = " ".join(tokens[short_ans['start_token']:short_ans['end_token']])
                        if text.strip():  # Добавляем только непустые контексты
                            short_contexts.append(text.strip())
    
    return long_contexts, short_contexts

def convert_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Конвертирует один элемент NQ в нужный формат.
    
    Args:
        item: Элемент из NQ датасета
        
    Returns:
        dict: Конвертированный элемент
    """
    # Извлекаем контексты
    long_contexts, short_contexts = extract_contexts(item)
    
    # Извлекаем URL и title из document_url
    url = item.get('document_url', '')
    title = ''
    if url:
        # Пытаемся извлечь title из URL (для Wikipedia)
        if 'wikipedia.org' in url:
            try:
                title = url.split('/')[-1].replace('_', ' ')
            except:
                title = ''
    
    # Создаем конвертированный элемент
    converted = {
        'question': item.get('question', ''),
        'answer': item.get('answer', []),  # Оставляем как массив
        'context': item.get('document_text', ''),  # Весь текст документа
        'long_context': long_contexts,  # Список long contexts
        'short_context': short_contexts,  # Список short contexts
        'id': str(item.get('example_id', '')),
        'url': url,
        'title': title
    }
    
    return converted

def convert_nq_data(input_file: str, output_file: str, max_items: int = None):
    """
    Конвертирует NQ данные из JSONL в JSON формат.
    
    Args:
        input_file: Путь к входному JSONL файлу
        output_file: Путь к выходному JSON файлу
        max_items: Максимальное количество элементов для обработки (None = все)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_file}")
    
    # Создаем директорию для выходного файла
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    converted_data = []
    processed = 0
    
    logger.info(f"Начинаем конвертацию из {input_file} в {output_file}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_items and processed >= max_items:
                break
                
            try:
                # Парсим JSON строку
                item = json.loads(line.strip())
                
                # Конвертируем элемент
                converted_item = convert_item(item)
                converted_data.append(converted_item)
                
                processed += 1
                
                if processed % 100 == 0:
                    logger.info(f"Обработано {processed} элементов...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Ошибка парсинга JSON в строке {line_num}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Ошибка обработки элемента в строке {line_num}: {e}")
                continue
    
    # Сохраняем результат
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Конвертация завершена. Обработано {processed} элементов.")
    logger.info(f"Результат сохранен в {output_file}")
    
    # Выводим статистику
    if converted_data:
        sample = converted_data[0]
        logger.info(f"Пример конвертированного элемента:")
        logger.info(f"  Question: {sample['question']}")
        logger.info(f"  Answer: {sample['answer']}")
        logger.info(f"  Long contexts: {len(sample['long_context'])}")
        logger.info(f"  Short contexts: {len(sample['short_context'])}")
        logger.info(f"  Context length: {len(sample['context'])} chars")

def main():
    parser = argparse.ArgumentParser(description='Конвертер данных Natural Questions')
    parser.add_argument('--input', '-i', required=True, help='Путь к входному JSONL файлу')
    parser.add_argument('--output', '-o', required=True, help='Путь к выходному JSON файлу')
    parser.add_argument('--max-items', '-m', type=int, help='Максимальное количество элементов для обработки')
    
    args = parser.parse_args()
    
    try:
        convert_nq_data(args.input, args.output, args.max_items)
    except Exception as e:
        logger.error(f"Ошибка конвертации: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
