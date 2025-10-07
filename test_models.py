#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест загрузки и работы с моделями.
"""

import sys
from pathlib import Path
import yaml
import logging

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.models import get_model

def test_model_loading():
    """Тестирует загрузку различных моделей."""
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Тестовые конфигурации
    test_configs = [
        {
            'name': 'smollm2_135m_test',
            'model_type': 'smollm2',
            'model_size': '135M',
            'model_path': 'HuggingFaceTB/SmolLM-135M',
            'max_length': 256,  # Уменьшено для теста
            'temperature': 0.7,
            'top_p': 0.9,
            'device': 'cpu',  # Используем CPU для теста
            'batch_size': 1,
            'use_flash_attention': False,
            'load_in_8bit': False,
            'gradient_checkpointing': False
        },
        {
            'name': 'qwen_0.6b_test',
            'model_type': 'qwen',
            'model_size': '0.6B',
            'model_path': 'unsloth/Qwen3-0.6B',
            'max_length': 256,
            'temperature': 0.7,
            'top_p': 0.9,
            'device': 'cpu',
            'batch_size': 1,
            'use_flash_attention': False,
            'load_in_8bit': False,
            'gradient_checkpointing': False
        }
    ]
    
    for config in test_configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Тестируем модель: {config['name']}")
        logger.info(f"Тип: {config['model_type']}, Размер: {config['model_size']}")
        logger.info(f"Путь: {config['model_path']}")
        logger.info(f"Устройство: {config['device']}")
        
        try:
            # Создаем модель
            logger.info("Загружаем модель...")
            model = get_model(config)
            
            # Проверяем размер модели
            model_size = model.get_model_size()
            logger.info(f"Размер модели: {model_size}")
            
            # Тестируем генерацию
            logger.info("Тестируем генерацию...")
            test_prompt = "What is the capital of France?"
            test_context = ["France is a country in Europe. Paris is the capital city of France."]
            
            response = model.generate(test_prompt, test_context)
            logger.info(f"Вопрос: {test_prompt}")
            logger.info(f"Контекст: {test_context[0]}")
            logger.info(f"Ответ модели: {response}")
            
            # Проверяем использование памяти
            memory_usage = model.get_memory_usage()
            logger.info(f"Использование памяти: {memory_usage}")
            
            logger.info("✅ Модель работает корректно!")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при работе с моделью {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"\n{'='*50}")
    logger.info("Тестирование завершено!")

def test_model_configs():
    """Тестирует загрузку конфигураций из файлов."""
    
    logger = logging.getLogger(__name__)
    config_dir = Path("configs/model")
    
    if not config_dir.exists():
        logger.warning("Директория configs/model не найдена")
        return
    
    config_files = list(config_dir.glob("*.yaml"))
    logger.info(f"Найдено {len(config_files)} конфигурационных файлов")
    
    for config_file in config_files:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"\nКонфигурация: {config_file.name}")
            logger.info(f"  Имя: {config.get('name')}")
            logger.info(f"  Тип: {config.get('model_type')}")
            logger.info(f"  Размер: {config.get('model_size')}")
            logger.info(f"  Путь: {config.get('model_path')}")
            logger.info(f"  Устройство: {config.get('device')}")
            logger.info(f"  Batch size: {config.get('batch_size')}")
            logger.info(f"  8-bit: {config.get('load_in_8bit', False)}")
            logger.info(f"  Gradient checkpointing: {config.get('gradient_checkpointing', False)}")
            
        except Exception as e:
            logger.error(f"Ошибка при чтении {config_file}: {e}")

if __name__ == '__main__':
    print("Тестирование работы с моделями")
    print("=" * 50)
    
    # Тестируем конфигурации
    test_model_configs()
    
    # Тестируем загрузку моделей (только если есть доступ к интернету)
    try:
        test_model_loading()
    except Exception as e:
        print(f"Ошибка при тестировании моделей: {e}")
        print("Возможно, нет доступа к HuggingFace или недостаточно памяти")
