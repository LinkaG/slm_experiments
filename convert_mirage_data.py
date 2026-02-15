#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конвертер данных MIRAGE в формат для slm_experiments.

Конвертирует MIRAGE/mirage/dataset.json в формат, совместимый с LocalMirageDataset.
Скачивает документы по doc_url и очищает их от HTML разметки.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urlparse
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Заголовки для запросов
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def clean_html(html_content: str, url: str) -> str:
    """
    Очищает HTML контент от разметки и извлекает текст.
    
    Args:
        html_content: HTML контент
        url: URL документа (для определения типа страницы)
        
    Returns:
        str: Очищенный текст
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Для Wikipedia страниц извлекаем основной контент
        if 'wikipedia.org' in url:
            # Удаляем навигацию, боковые панели, футеры
            for element in soup.find_all(['nav', 'aside', 'footer', 'header', 'script', 'style']):
                element.decompose()
            
            # Находим основной контент статьи
            content_div = soup.find('div', {'id': 'mw-content-text'})
            if content_div:
                # Удаляем инфобоксы и другие боковые элементы
                for element in content_div.find_all(['div', 'table'], class_=re.compile(r'infobox|sidebar|navbox')):
                    element.decompose()
                
                # Извлекаем текст
                text = content_div.get_text(separator=' ', strip=True)
            else:
                # Fallback: извлекаем весь текст из body
                body = soup.find('body')
                if body:
                    text = body.get_text(separator=' ', strip=True)
                else:
                    text = soup.get_text(separator=' ', strip=True)
        else:
            # Для других сайтов просто извлекаем текст из body
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
        
        # Очищаем множественные пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
        
    except Exception as e:
        logger.warning(f"Ошибка очистки HTML для {url}: {e}")
        # Fallback: возвращаем сырой текст без HTML тегов
        text = re.sub(r'<[^>]+>', '', html_content)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

def download_document(url: str, timeout: int = 30, retries: int = 3) -> Optional[str]:
    """
    Скачивает документ по URL и возвращает очищенный текст.
    
    Args:
        url: URL документа
        timeout: Таймаут запроса в секундах
        retries: Количество попыток при ошибке
        
    Returns:
        str: Очищенный текст документа или None при ошибке
    """
    for attempt in range(retries):
        try:
            logger.debug(f"Скачиваем документ: {url} (попытка {attempt + 1}/{retries})")
            response = requests.get(url, headers=HEADERS, timeout=timeout)
            response.raise_for_status()
            
            # Определяем тип контента
            content_type = response.headers.get('Content-Type', '').lower()
            if 'html' in content_type or url.endswith('.html') or '?' in url:
                # HTML контент - очищаем
                return clean_html(response.text, url)
            else:
                # Текстовый контент - возвращаем как есть
                return response.text.strip()
                
        except requests.exceptions.Timeout:
            logger.warning(f"Таймаут при скачивании {url} (попытка {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Экспоненциальная задержка
        except requests.exceptions.RequestException as e:
            logger.warning(f"Ошибка при скачивании {url}: {e} (попытка {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Неожиданная ошибка при скачивании {url}: {e}")
            return None
    
    logger.error(f"Не удалось скачать документ после {retries} попыток: {url}")
    return None

def convert_item(
    item: Dict[str, Any], 
    download_docs: bool = True, 
    cache_dir: Optional[Path] = None,
    oracle_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Конвертирует один элемент MIRAGE в нужный формат.
    
    Args:
        item: Элемент из MIRAGE датасета
        download_docs: Скачивать ли документы по doc_url
        cache_dir: Директория для кеширования документов
        oracle_data: Словарь с данными из oracle.json (ключ - query_id)
        
    Returns:
        dict: Конвертированный элемент
    """
    # Извлекаем основные поля
    question = item.get('query', '')
    answers = item.get('answer', [])
    doc_url = item.get('doc_url', '')
    doc_name = item.get('doc_name', '')
    query_id = item.get('query_id', '')
    source = item.get('source', '')
    
    num_doc_labels = item.get('num_doc_labels', 0)
    
    # Извлекаем long_answer (doc_chunk) из oracle.json если доступен
    long_answer = None
    if oracle_data and query_id in oracle_data:
        oracle_item = oracle_data[query_id]
        long_answer = oracle_item.get('doc_chunk')
    
    # Преобразуем ответы: если список, берем первый или объединяем
    if isinstance(answers, list):
        if len(answers) > 0:
            answer = answers[0]  # Берем первый ответ как основной
        else:
            answer = None
    else:
        answer = answers
    
    # Скачиваем документ если нужно
    context = None
    if download_docs and doc_url:
        # Проверяем кеш
        if cache_dir:
            cache_file = cache_dir / f"{query_id}.txt"
            if cache_file.exists():
                logger.debug(f"Загружаем из кеша: {cache_file}")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    context = f.read()
            else:
                context = download_document(doc_url)
                if context and cache_dir:
                    # Сохраняем в кеш
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(context)
        else:
            context = download_document(doc_url)
    
    # Создаем конвертированный элемент в формате, совместимом с SimpleQA и NQ
    # long_answer добавляем как список для совместимости с NQ форматом
    long_context = [long_answer] if long_answer else []
    
    converted = {
        'question': question,
        'answer': answer,
        'context': context,  # Может быть None если документ не скачался
        'metadata': {
            'id': query_id,
            'url': doc_url,
            'title': doc_name,
            'source': source,
            'query_id': query_id,
            'doc_name': doc_name,
            'all_answers': answers,  # Сохраняем все варианты ответов
            'num_doc_labels': num_doc_labels,
            'long_answer': long_answer,  # doc_chunk из oracle.json
            'long_context': long_context  # Список для совместимости с NQ
        }
    }
    
    return converted

def convert_mirage_data(
    input_file: str, 
    output_file: str, 
    max_items: int = None,
    download_docs: bool = True,
    cache_dir: Optional[str] = None,
    delay: float = 0.5,
    oracle_file: Optional[str] = None
):
    """
    Конвертирует MIRAGE данные из JSON в нужный формат.
    
    Args:
        input_file: Путь к входному JSON файлу
        output_file: Путь к выходному JSON файлу
        max_items: Максимальное количество элементов для обработки (None = все)
        download_docs: Скачивать ли документы по doc_url
        cache_dir: Директория для кеширования документов
        delay: Задержка между запросами в секундах (для избежания rate limiting)
        oracle_file: Путь к файлу oracle.json с doc_chunk данными
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_file}")
    
    # Создаем директорию для выходного файла
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Создаем директорию для кеша если указана
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
    
    # Загружаем oracle.json если указан
    oracle_data = None
    if oracle_file:
        oracle_path = Path(oracle_file)
        if oracle_path.exists():
            logger.info(f"Загружаем oracle.json из {oracle_file}...")
            with open(oracle_path, 'r', encoding='utf-8') as f:
                oracle_data = json.load(f)
            logger.info(f"Загружено {len(oracle_data)} записей из oracle.json")
        else:
            logger.warning(f"Файл oracle.json не найден: {oracle_file}")
    
    converted_data = []
    processed = 0
    failed_downloads = 0
    
    logger.info(f"Начинаем конвертацию из {input_file} в {output_file}")
    logger.info(f"Скачивание документов: {'включено' if download_docs else 'отключено'}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_items = len(data)
    if max_items:
        total_items = min(total_items, max_items)
    
    logger.info(f"Всего элементов для обработки: {total_items}")
    
    for idx, item in enumerate(data):
        if max_items and processed >= max_items:
            break
        
        try:
            # Конвертируем элемент
            converted_item = convert_item(
                item, 
                download_docs=download_docs, 
                cache_dir=cache_path,
                oracle_data=oracle_data
            )
            
            # Проверяем наличие контекста если скачивание включено
            if download_docs and not converted_item.get('context'):
                failed_downloads += 1
                logger.warning(f"Не удалось скачать документ для элемента {idx + 1}: {converted_item.get('url')}")
            
            converted_data.append(converted_item)
            processed += 1
            
            if processed % 100 == 0:
                logger.info(f"Обработано {processed}/{total_items} элементов... (неудачных загрузок: {failed_downloads})")
            
            # Задержка между запросами для избежания rate limiting
            if download_docs and delay > 0 and idx < len(data) - 1:
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Ошибка обработки элемента {idx + 1}: {e}")
            continue
    
    # Сохраняем результат
    logger.info(f"Сохраняем результат в {output_file}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Конвертация завершена. Обработано {processed} элементов.")
    logger.info(f"Неудачных загрузок документов: {failed_downloads}")
    logger.info(f"Результат сохранен в {output_file}")
    
    # Выводим статистику
    if converted_data:
        sample = converted_data[0]
        logger.info(f"\nПример конвертированного элемента:")
        logger.info(f"  Question: {sample['question']}")
        logger.info(f"  Answer: {sample['answer']}")
        context_len = len(sample.get('context', '')) if sample.get('context') else 0
        logger.info(f"  Context length: {context_len} chars")
        logger.info(f"  URL: {sample.get('url', 'N/A')}")
        
        # Статистика по контекстам
        with_context = sum(1 for item in converted_data if item.get('context'))
        with_long_answer = sum(1 for item in converted_data 
                             if item.get('metadata', {}).get('long_answer'))
        logger.info(f"\nСтатистика:")
        logger.info(f"  Всего элементов: {len(converted_data)}")
        logger.info(f"  С контекстом: {with_context}")
        logger.info(f"  Без контекста: {len(converted_data) - with_context}")
        logger.info(f"  С long_answer (doc_chunk): {with_long_answer}")

def main():
    parser = argparse.ArgumentParser(description='Конвертер данных MIRAGE')
    parser.add_argument('--input', '-i', 
                       default='MIRAGE/mirage/dataset.json',
                       help='Путь к входному JSON файлу')
    parser.add_argument('--output', '-o', 
                       default='data/mirage/mirage_converted.json',
                       help='Путь к выходному JSON файлу')
    parser.add_argument('--max-items', '-m', type=int, 
                       help='Максимальное количество элементов для обработки')
    parser.add_argument('--no-download', action='store_true',
                       help='Не скачивать документы по doc_url')
    parser.add_argument('--cache-dir', type=str,
                       default='.cache/mirage_docs',
                       help='Директория для кеширования документов')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Задержка между запросами в секундах (по умолчанию 0.5)')
    parser.add_argument('--oracle-file', type=str,
                       default='MIRAGE/mirage/oracle.json',
                       help='Путь к файлу oracle.json с doc_chunk данными')
    
    args = parser.parse_args()
    
    try:
        convert_mirage_data(
            args.input, 
            args.output, 
            max_items=args.max_items,
            download_docs=not args.no_download,
            cache_dir=args.cache_dir,
            delay=args.delay,
            oracle_file=args.oracle_file
        )
    except Exception as e:
        logger.error(f"Ошибка конвертации: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

