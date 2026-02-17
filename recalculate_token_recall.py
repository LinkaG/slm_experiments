#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для пересчета метрики token_recall для экспериментов в outputs_2
и обновления результатов в ClearML.

Использование:

1. Через Docker сеть (рекомендуется, если ClearML доступен по Docker сети):
    ./run_recalculate_token_recall.sh
    ./run_recalculate_token_recall.sh --dry-run  # без обновления файлов
    ./run_recalculate_token_recall.sh --no-clearml  # без обновления ClearML
    ./run_recalculate_token_recall.sh --experiment qwen_0.6b_local_nq_full_no_context

2. Локально (если ClearML доступен напрямую):
    poetry run python recalculate_token_recall.py
    poetry run python recalculate_token_recall.py --dry-run
    poetry run python recalculate_token_recall.py --no-clearml
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from statistics import mean

from clearml import Task, Logger

from src.experiment.metrics import TokenRecallCalculator
from src.utils.clearml_config import setup_clearml_environment

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_clearml_task(task_name: str, project_name: str = "slm-experiments", 
                     task_id: Optional[str] = None, tags: Optional[List[str]] = None) -> Optional[Task]:
    """
    Находит существующую задачу ClearML по имени, ID или тегам.
    
    Args:
        task_name: Название задачи
        project_name: Название проекта
        task_id: ID задачи (если указан, используется напрямую)
        tags: Список тегов для поиска
        
    Returns:
        Task объект или None если не найдено
    """
    # Если указан task_id, используем его напрямую
    if task_id:
        try:
            task = Task.get_task(task_id=task_id)
            logger.info(f"  ✅ Задача найдена по ID: {task_id} (название: {task.name})")
            return task
        except Exception as e:
            logger.error(f"  ❌ Не удалось загрузить задачу по ID {task_id}: {e}")
            return None
    try:
        # Парсим конфигурационный файл и устанавливаем переменные окружения
        # Это нужно для правильной работы ClearML SDK в Docker контейнере
        config_path = Path.home() / ".clearml.conf"
        if config_path.exists():
            try:
                import re
                with open(config_path, 'r') as f:
                    content = f.read()
                    
                    # Извлекаем значения из HOCON конфигурации
                    api_match = re.search(r'api_server:\s*([^\s\n]+)', content)
                    web_match = re.search(r'web_server:\s*([^\s\n]+)', content)
                    files_match = re.search(r'files_server:\s*([^\s\n]+)', content)
                    
                    if api_match:
                        os.environ['CLEARML_API_HOST'] = api_match.group(1)
                    if web_match:
                        os.environ['CLEARML_WEB_HOST'] = web_match.group(1)
                    if files_match:
                        os.environ['CLEARML_FILES_HOST'] = files_match.group(1)
                    
                    # Ищем credentials (access_key и secret_key)
                    access_key_match = re.search(r'"access_key"\s*=\s*"([^"]+)"', content)
                    secret_key_match = re.search(r'"secret_key"\s*=\s*"([^"]+)"', content)
                    
                    if access_key_match:
                        os.environ['CLEARML_API_ACCESS_KEY'] = access_key_match.group(1)
                    if secret_key_match:
                        os.environ['CLEARML_API_SECRET_KEY'] = secret_key_match.group(1)
            except Exception as e:
                logger.debug(f"  ⚠️  Не удалось распарсить конфигурацию: {e}")
        
        # Пробуем разные способы поиска задачи
        
        # 1. Поиск по точному имени
        logger.info(f"  🔍 Ищем задачу по точному имени: '{task_name}' в проекте '{project_name}'")
        try:
            task_ids = Task.query_tasks(
                project_name=project_name,
                task_name=task_name
            )
            logger.info(f"  📊 Task.query_tasks вернул: {len(task_ids) if task_ids else 0} задач")
        except Exception as e:
            logger.error(f"  ❌ Ошибка при поиске по имени: {e}")
            task_ids = []
        
        # 2. Если не найдено, пробуем найти все задачи в проекте и фильтровать вручную
        if not task_ids:
            logger.info(f"  🔍 Точное совпадение не найдено, ищем все задачи в проекте...")
            try:
                all_task_ids = Task.query_tasks(project_name=project_name)
                logger.info(f"  📊 Всего задач в проекте: {len(all_task_ids) if all_task_ids else 0}")
                
                if all_task_ids:
                    matching_ids = []
                    # Показываем примеры найденных задач для отладки
                    example_names = []
                    # Проверяем первые 200 задач (на случай если их много)
                    for tid in all_task_ids[:200]:
                        try:
                            t = Task.get_task(task_id=tid)
                            # Собираем примеры имен для отладки
                            if len(example_names) < 5:
                                example_names.append(t.name)
                            
                            # Точное совпадение имени
                            if t.name == task_name:
                                matching_ids.append(tid)
                                logger.info(f"  ✅ Найдено точное совпадение: ID={tid}, name='{t.name}'")
                        except Exception as e:
                            logger.debug(f"    Не удалось загрузить задачу {tid}: {e}")
                            continue
                    
                    # Показываем примеры найденных имен задач
                    if example_names and not matching_ids:
                        logger.info(f"  📋 Примеры имен задач в проекте (первые 5):")
                        for i, name in enumerate(example_names, 1):
                            logger.info(f"    {i}. '{name}'")
                        logger.info(f"  🔍 Ищем: '{task_name}'")
                        logger.info(f"  💡 Проверьте, совпадает ли имя точно (включая регистр и все символы)")
                    
                    if matching_ids:
                        task_ids = matching_ids
                        logger.info(f"  ✅ Найдено {len(matching_ids)} задач с точным совпадением имени")
            except Exception as e:
                logger.error(f"  ❌ Ошибка при поиске всех задач: {e}")
        
        # 3. Если все еще не найдено и есть теги, пробуем поиск по тегам
        if not task_ids and tags:
            logger.info(f"  🔍 Пробуем найти задачу по тегам: {tags}")
            try:
                task_ids = Task.query_tasks(
                    project_name=project_name,
                    tags=tags
                )
                logger.info(f"  📊 Поиск по тегам вернул: {len(task_ids) if task_ids else 0} задач")
            except Exception as e:
                logger.error(f"  ❌ Ошибка при поиске по тегам: {e}")
        
        if not task_ids:
            logger.warning(f"  ⚠️  Задача '{task_name}' не найдена в проекте '{project_name}'")
            logger.info(f"  💡 Попробуйте указать task_id вручную через параметр --task-id")
            logger.info(f"  💡 Или проверьте, что проект называется именно '{project_name}'")
            return None
        
        # Если найдено несколько задач, берем самую свежую (первую в списке обычно самая свежая)
        # Или можно перебрать все и выбрать по дате создания
        best_task = None
        best_created = None
        
        logger.info(f"  📋 Обрабатываем {len(task_ids)} найденных задач...")
        for task_id in task_ids:
            try:
                task = Task.get_task(task_id=task_id)
                logger.info(f"  📝 Загружена задача: ID={task_id}, name='{task.name}', project='{task.project}'")
                
                # Проверяем, что это действительно нужная задача
                if task.name == task_name:
                    logger.info(f"  ✅ Имя задачи совпадает точно!")
                    # Берем самую свежую задачу (используем created_at или created, если доступно)
                    task_created = None
                    if hasattr(task, 'created_at'):
                        task_created = task.created_at
                    elif hasattr(task, 'created'):
                        task_created = task.created
                    elif hasattr(task, 'data') and hasattr(task.data, 'created'):
                        task_created = task.data.created
                    
                    if best_created is None or (task_created and (task_created > best_created)):
                        best_task = task
                        best_created = task_created
                    elif task_created is None:
                        # Если дата создания недоступна, просто используем эту задачу
                        best_task = task
                else:
                    logger.warning(f"  ⚠️  Имя задачи не совпадает: ожидали '{task_name}', получили '{task.name}'")
                    logger.info(f"  💡 Попробуем использовать эту задачу, так как она была найдена по запросу")
                    # Если это единственная найденная задача, используем её даже если имя немного отличается
                    if len(task_ids) == 1:
                        logger.info(f"  ✅ Используем найденную задачу (единственная в результате поиска)")
                        best_task = task
            except Exception as e:
                logger.error(f"  ❌ Не удалось загрузить задачу {task_id}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        if best_task:
            logger.info(f"  ✅ Найдена задача ClearML: {best_task.id} (название: {best_task.name})")
            return best_task
        else:
            logger.warning(f"  ⚠️  Не удалось найти подходящую задачу '{task_name}'")
            return None
            
    except Exception as e:
        logger.error(f"  ❌ Ошибка при поиске задачи ClearML: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def recalculate_token_recall_for_experiment(
    experiment_dir: Path,
    dry_run: bool = False,
    update_clearml: bool = True,
    task_id: Optional[str] = None
) -> Dict:
    """
    Пересчитывает token_recall для одного эксперимента.
    
    Args:
        experiment_dir: Директория с результатами эксперимента
        dry_run: Если True, не сохраняет изменения
        update_clearml: Если True, обновляет метрики в ClearML
        
    Returns:
        Словарь с результатами пересчета
    """
    experiment_name = experiment_dir.name
    logger.info(f"\n{'='*60}")
    logger.info(f"📁 Обработка эксперимента: {experiment_name}")
    logger.info(f"{'='*60}")
    
    predictions_file = experiment_dir / "predictions.json"
    results_file = experiment_dir / "results.json"
    
    if not predictions_file.exists():
        logger.warning(f"  ⚠️  Файл predictions.json не найден, пропускаем")
        return {"status": "skipped", "reason": "predictions.json not found"}
    
    if not results_file.exists():
        logger.warning(f"  ⚠️  Файл results.json не найден, пропускаем")
        return {"status": "skipped", "reason": "results.json not found"}
    
    # Загружаем predictions
    logger.info("  📖 Загружаем predictions.json...")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)
    
    predictions = predictions_data.get('predictions', [])
    if not predictions:
        logger.warning(f"  ⚠️  Нет предсказаний в файле, пропускаем")
        return {"status": "skipped", "reason": "no predictions"}
    
    logger.info(f"  📊 Найдено {len(predictions)} предсказаний")
    
    # Создаем калькулятор метрик
    logger.info("  🔧 Инициализируем TokenRecallCalculator...")
    calculator = TokenRecallCalculator()
    
    # Пересчитываем token_recall для каждого предсказания
    logger.info("  🔄 Пересчитываем token_recall...")
    old_recalls = []
    new_recalls = []
    
    for i, pred in enumerate(predictions):
        predicted_answer = pred.get('predicted_answer', '')
        ground_truth = pred.get('ground_truth', [])
        
        if not predicted_answer or not ground_truth:
            continue
        
        old_recall = pred.get('token_recall', 0.0)
        old_recalls.append(old_recall)
        
        # Пересчитываем recall
        new_recall = calculator.calculate_recall(predicted_answer, ground_truth)
        new_recalls.append(new_recall)
        
        # Обновляем значение в словаре
        pred['token_recall'] = new_recall
        
        # Логируем прогресс периодически
        if (i + 1) % 100 == 0:
            logger.info(f"    Обработано {i + 1}/{len(predictions)} предсказаний...")
    
    if not new_recalls:
        logger.warning(f"  ⚠️  Не удалось пересчитать ни одного recall, пропускаем")
        return {"status": "skipped", "reason": "no valid predictions"}
    
    # Вычисляем средние значения
    old_mean = mean(old_recalls) if old_recalls else 0.0
    new_mean = mean(new_recalls) if new_recalls else 0.0
    
    logger.info(f"  📈 Старое среднее token_recall: {old_mean:.6f}")
    logger.info(f"  📈 Новое среднее token_recall: {new_mean:.6f}")
    logger.info(f"  📊 Разница: {new_mean - old_mean:+.6f}")
    
    # Обновляем results.json
    logger.info("  📝 Обновляем results.json...")
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    old_result_recall = results_data.get('token_recall', 0.0)
    results_data['token_recall'] = new_mean
    results_data['num_examples'] = len(new_recalls)
    
    logger.info(f"  📈 Старое значение в results.json: {old_result_recall:.6f}")
    logger.info(f"  📈 Новое значение в results.json: {new_mean:.6f}")
    
    # Сохраняем изменения
    if not dry_run:
        logger.info("  💾 Сохраняем обновленные файлы...")
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_data, f, indent=2, ensure_ascii=False)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info("  ✅ Файлы успешно сохранены")
    else:
        logger.info("  🔍 DRY RUN: файлы не сохранены")
    
    # Обновляем метрики в ClearML
    clearml_task = None
    if update_clearml and not dry_run:
        logger.info("  🔗 Обновляем метрики в ClearML...")
        try:
            setup_clearml_environment()
            
            # Извлекаем теги из имени эксперимента для поиска
            # Формат: model_dataset_mode -> теги: [model, dataset, mode]
            parts = experiment_name.split('_')
            tags = []
            if len(parts) >= 2:
                tags.append(parts[0])  # модель
                if len(parts) >= 3:
                    tags.append('_'.join(parts[1:-1]))  # датасет (может быть с подчеркиваниями)
                    tags.append(parts[-1])  # режим (no_context и т.д.)
            
            clearml_task = find_clearml_task(
                task_name=experiment_name,
                task_id=task_id,
                tags=tags if tags else None
            )
            
            if clearml_task:
                logger.info(f"  📤 Загружаем обновленные метрики в ClearML...")
                logger.info(f"  📋 Статус задачи: {clearml_task.status}")
                
                # Для существующей задачи нужно использовать её логгер
                # Сначала пробуем получить логгер через Task.get_logger()
                try:
                    clearml_logger = clearml_task.get_logger()
                except Exception as e:
                    logger.warning(f"  ⚠️  Не удалось получить логгер задачи: {e}")
                    # Fallback: используем текущий логгер (может не сработать для закрытых задач)
                    clearml_logger = Logger.current_logger()
                
                # Обновляем single value метрику
                try:
                    clearml_logger.report_single_value(
                        name="token_recall",
                        value=new_mean
                    )
                    
                    # Также логируем текстовое сообщение об обновлении
                    clearml_logger.report_text(
                        f"🔄 Метрика token_recall была пересчитана и обновлена: {new_mean:.6f} "
                        f"(старое значение: {old_result_recall:.6f})"
                    )
                    
                    logger.info(f"  ✅ Метрики обновлены в ClearML")
                    logger.info(f"  🔗 Ссылка: {clearml_task.get_output_log_web_page()}")
                except Exception as e:
                    logger.error(f"  ❌ Не удалось обновить метрики в ClearML: {e}")
                    logger.warning(f"  💡 Возможно, задача закрыта. Попробуйте открыть её вручную или используйте API")
            else:
                logger.warning(f"  ⚠️  Задача ClearML не найдена, метрики не обновлены")
                
        except Exception as e:
            logger.error(f"  ❌ Ошибка при обновлении ClearML: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    return {
        "status": "success",
        "experiment": experiment_name,
        "num_predictions": len(new_recalls),
        "old_mean": old_mean,
        "new_mean": new_mean,
        "difference": new_mean - old_mean,
        "old_result_recall": old_result_recall,
        "clearml_updated": clearml_task is not None
    }


def main():
    parser = argparse.ArgumentParser(
        description='Пересчет метрики token_recall для экспериментов в outputs_2'
    )
    parser.add_argument(
        '--outputs-dir',
        type=str,
        default='outputs',
        help='Директория с результатами экспериментов (по умолчанию: outputs_2)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Режим проверки без сохранения изменений'
    )
    parser.add_argument(
        '--no-clearml',
        action='store_true',
        help='Не обновлять метрики в ClearML'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Обработать только указанный эксперимент (имя папки)'
    )
    parser.add_argument(
        '--task-id',
        type=str,
        default=None,
        help='ID задачи ClearML для обновления (если не указан, будет поиск по имени)'
    )
    
    args = parser.parse_args()
    
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        logger.error(f"❌ Директория не найдена: {outputs_dir}")
        return 1
    
    logger.info(f"🔍 Поиск экспериментов в {outputs_dir}")
    
    # Находим все папки с экспериментами
    experiment_dirs = [
        d for d in outputs_dir.iterdir()
        if d.is_dir() and (d / "predictions.json").exists()
    ]
    
    if args.experiment:
        experiment_dirs = [d for d in experiment_dirs if d.name == args.experiment]
        if not experiment_dirs:
            logger.error(f"❌ Эксперимент '{args.experiment}' не найден")
            return 1
    
    if not experiment_dirs:
        logger.warning(f"⚠️  Не найдено экспериментов для обработки")
        return 0
    
    logger.info(f"📊 Найдено {len(experiment_dirs)} экспериментов для обработки")
    
    if args.dry_run:
        logger.info("🔍 РЕЖИМ ПРОВЕРКИ (DRY RUN) - файлы не будут изменены")
    
    # Обрабатываем каждый эксперимент
    results = []
    for experiment_dir in sorted(experiment_dirs):
        try:
            result = recalculate_token_recall_for_experiment(
                experiment_dir=experiment_dir,
                dry_run=args.dry_run,
                update_clearml=not args.no_clearml,
                task_id=args.task_id
            )
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Ошибка при обработке {experiment_dir.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results.append({
                "status": "error",
                "experiment": experiment_dir.name,
                "error": str(e)
            })
    
    # Выводим итоговую статистику
    logger.info("\n" + "="*60)
    logger.info("📊 ИТОГОВАЯ СТАТИСТИКА")
    logger.info("="*60)
    
    successful = [r for r in results if r.get("status") == "success"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    errors = [r for r in results if r.get("status") == "error"]
    
    logger.info(f"✅ Успешно обработано: {len(successful)}")
    logger.info(f"⏭️  Пропущено: {len(skipped)}")
    logger.info(f"❌ Ошибок: {len(errors)}")
    
    if successful:
        logger.info("\n📈 Изменения метрик:")
        for result in successful:
            logger.info(
                f"  {result['experiment']}: "
                f"{result['old_mean']:.6f} → {result['new_mean']:.6f} "
                f"({result['difference']:+.6f})"
            )
    
    if errors:
        logger.info("\n❌ Ошибки:")
        for result in errors:
            logger.info(f"  {result['experiment']}: {result.get('error', 'unknown error')}")
    
    return 0 if not errors else 1


if __name__ == "__main__":
    exit(main())

