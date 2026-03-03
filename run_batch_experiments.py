#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для автоматического запуска серии экспериментов с управлением GPU памятью.
Запускает все комбинации моделей и датасетов с максимальным параллелизмом.

Использование:
    # Запуск всех экспериментов (все модели × все датасеты)
    poetry run python run_batch_experiments.py
    
    # Запуск с конкретными моделями и датасетами
    poetry run python run_batch_experiments.py --models qwen_0.6b qwen_1.7b --datasets local_simple_qa
    
    # Ограничение параллелизма
    poetry run python run_batch_experiments.py --max-parallel 2
    
    # Без ClearML логирования
    poetry run python run_batch_experiments.py --no-clearml
    
    # Oracle эксперименты в отдельный проект ClearML и папку output_2
    poetry run python run_batch_experiments.py --experiment-mode oracle_context --clearml-project oracle --output-dir output_2
    
    # Oracle long context (только NQ и MIRAGE, simple_qa исключён - нет long_answer)
    poetry run python run_batch_experiments.py --experiment-mode oracle_long_context --clearml-project oracle --output-dir output_3
    
    # Или если активировано окружение Poetry (poetry shell):
    python run_batch_experiments.py

Особенности:
    - MinIO bucket = имя конфига experiment_mode (no_context, oracle_context, oracle_long_context)
    - Автоматическое определение доступных GPU
    - Мониторинг памяти GPU через nvidia-smi
    - Максимальный параллелизм с учетом доступной памяти
    - Автоматический retry при ошибках (по умолчанию 3 попытки)
    - Логирование прогресса в ClearML
    - Оценка требований к памяти для каждой модели
"""

import subprocess
import json
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from queue import Queue, Empty as QueueEmpty
from omegaconf import OmegaConf
import yaml

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('batch_experiments.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentTask:
    """Задача эксперимента."""
    model: str
    dataset: str
    experiment_mode: str
    estimated_memory_gb: float
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"  # pending, running, completed, failed
    task_id: str = ""  # Уникальный ID задачи
    gpu_id: Optional[int] = None  # GPU, на которой выполняется задача
    failure_reason: Optional[str] = None  # Причина неудачи (для логирования)


class GPUMonitor:
    """Мониторинг GPU памяти."""
    
    def __init__(self):
        self.gpu_count = self._get_gpu_count()
        self.gpu_locks = [threading.Lock() for _ in range(self.gpu_count)]  # Блокировки для каждой GPU
        self.gpu_reservations = {}  # Словарь: gpu_id -> task_id
        logger.info(f"🔍 Обнаружено GPU: {self.gpu_count}")
    
    def _get_gpu_count(self) -> int:
        """Получить количество доступных GPU."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--list-gpus'],
                capture_output=True,
                text=True,
                check=True
            )
            return len(result.stdout.strip().split('\n'))
        except Exception as e:
            logger.warning(f"⚠️  Не удалось определить количество GPU: {e}")
            return 0
    
    def get_free_memory_per_gpu(self) -> List[float]:
        """
        Получить свободную память для каждой GPU в GB.
        Учитывает реальное использование GPU всеми процессами (включая других пользователей).
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            # Конвертируем из MB в GB
            free_memories = [float(line.strip()) / 1024.0 for line in result.stdout.strip().split('\n') if line.strip()]
            return free_memories
        except Exception as e:
            logger.error(f"❌ Ошибка при получении информации о GPU памяти: {e}")
            return [0.0] * self.gpu_count
    
    def is_gpu_actually_free(self, gpu_id: int, required_memory_gb: float, reserved_memory_gb: float = 2.0) -> bool:
        """
        Проверить, действительно ли GPU свободна и доступна для использования.
        Учитывает реальное использование GPU другими процессами (включая других пользователей).
        
        Args:
            gpu_id: ID GPU для проверки
            required_memory_gb: Требуемая память в GB
            reserved_memory_gb: Резерв памяти для системы
        
        Returns:
            True если GPU действительно свободна и доступна
        """
        # Проверяем, что GPU не зарезервирована нашим скриптом
        if gpu_id in self.gpu_reservations:
            return False
        
        # Проверяем реальное использование GPU через nvidia-smi
        # Это учитывает процессы всех пользователей
        free_memories = self.get_free_memory_per_gpu()
        
        if gpu_id >= len(free_memories):
            return False
        
        free_memory = free_memories[gpu_id]
        
        # Проверяем, что достаточно свободной памяти
        if free_memory < (required_memory_gb + reserved_memory_gb):
            logger.debug(f"GPU {gpu_id} занята другими процессами: свободно {free_memory:.1f}GB, требуется {required_memory_gb + reserved_memory_gb:.1f}GB")
            return False
        
        return True
    
    def get_total_memory_per_gpu(self) -> List[float]:
        """Получить общую память для каждой GPU в GB."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            # Конвертируем из MB в GB
            total_memories = [float(line.strip()) / 1024.0 for line in result.stdout.strip().split('\n') if line.strip()]
            return total_memories
        except Exception as e:
            logger.error(f"❌ Ошибка при получении информации о GPU памяти: {e}")
            return [0.0] * self.gpu_count
    
    def find_available_gpu(self, required_memory_gb: float, reserved_memory_gb: float = 2.0, task_id: Optional[str] = None, enforce_exclusive: bool = True) -> Optional[int]:
        """
        Найти доступную GPU с достаточным количеством свободной памяти.
        Резервирует GPU для задачи, чтобы избежать конфликтов между воркерами.
        
        ВАЖНО: По умолчанию enforce_exclusive=True гарантирует, что одна GPU используется
        только одним экспериментом одновременно. Это необходимо для корректного логирования
        мощности GPU - nvidia-smi показывает общее потребление всей GPU, а не отдельного процесса.
        
        Args:
            required_memory_gb: Требуемая память в GB
            reserved_memory_gb: Резерв памяти для системы (по умолчанию 2GB)
            task_id: Уникальный ID задачи для резервирования
            enforce_exclusive: Если True, гарантирует эксклюзивное использование GPU
                              (одна GPU = один эксперимент). Необходимо для корректного логирования мощности.
        
        Returns:
            Индекс GPU или None если нет доступной
        """
        # Проверяем каждую GPU
        for gpu_idx in range(self.gpu_count):
            # КРИТИЧНО: Проверяем, что GPU не зарезервирована нашим скриптом
            # Это гарантирует, что одна GPU используется только одним экспериментом
            if gpu_idx in self.gpu_reservations:
                continue
            
            # Проверяем реальное использование GPU (учитывает процессы других пользователей)
            if not self.is_gpu_actually_free(gpu_idx, required_memory_gb, reserved_memory_gb):
                continue
            
            # Дополнительная проверка: если enforce_exclusive=True, проверяем что GPU полностью свободна
            # (не используется другими процессами, даже если есть свободная память)
            if enforce_exclusive:
                # Проверяем, что GPU не используется другими процессами
                # Если используется память другими процессами, пропускаем эту GPU
                free_memories = self.get_free_memory_per_gpu()
                total_memories = self.get_total_memory_per_gpu()
                
                if gpu_idx < len(free_memories) and gpu_idx < len(total_memories):
                    free_mem = free_memories[gpu_idx]
                    total_mem = total_memories[gpu_idx]
                    used_mem = total_mem - free_mem
                    
                    # Если GPU используется другими процессами (использовано больше 5% памяти),
                    # пропускаем её для гарантии эксклюзивности
                    if used_mem > total_mem * 0.05:  # Более 5% памяти используется
                        logger.debug(f"⏭️  GPU {gpu_idx} пропущена: используется другими процессами ({used_mem:.1f}GB/{total_mem:.1f}GB)")
                        continue
            
            # Пытаемся заблокировать GPU
            if self.gpu_locks[gpu_idx].acquire(blocking=False):
                try:
                    # Двойная проверка (double-checked locking)
                    # Проверяем еще раз, что GPU не зарезервирована и реально свободна
                    if gpu_idx not in self.gpu_reservations and self.is_gpu_actually_free(gpu_idx, required_memory_gb, reserved_memory_gb):
                        if task_id:
                            self.gpu_reservations[gpu_idx] = task_id
                        logger.debug(f"✅ GPU {gpu_idx} зарезервирована эксклюзивно для задачи {task_id}")
                        return gpu_idx
                finally:
                    self.gpu_locks[gpu_idx].release()
        
        return None
    
    def release_gpu(self, gpu_id: int, task_id: Optional[str] = None):
        """
        Освободить GPU после завершения задачи.
        
        Args:
            gpu_id: ID GPU для освобождения
            task_id: ID задачи (для проверки)
        """
        if gpu_id in self.gpu_reservations:
            if task_id is None or self.gpu_reservations[gpu_id] == task_id:
                del self.gpu_reservations[gpu_id]
                logger.debug(f"🔓 GPU {gpu_id} освобождена")


class BatchExperimentRunner:
    """Менеджер для запуска пакетных экспериментов."""
    
    def __init__(
        self,
        models: List[str],
        datasets: List[str],
        experiment_mode: str = "no_context",
        max_parallel: Optional[int] = None,
        retry_count: int = 3,
        use_clearml: bool = True,
        clearml_project: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        self.models = models
        self.datasets = datasets
        self.experiment_mode = experiment_mode
        self.clearml_project = clearml_project
        self.output_dir = output_dir
        self.max_retries = retry_count
        self.use_clearml = use_clearml
        
        self.gpu_monitor = GPUMonitor()
        self.gpu_count = self.gpu_monitor.gpu_count
        
        # Максимальное количество параллельных экспериментов
        # Если не указано, используем количество GPU (чтобы использовать все GPU параллельно)
        # ВАЖНО: Для корректного логирования мощности GPU рекомендуется использовать
        # max_parallel <= gpu_count, чтобы каждая GPU использовалась только одним экспериментом
        self.max_parallel = max_parallel or self.gpu_count
        
        # Гарантируем эксклюзивное использование GPU для корректного логирования мощности
        # Если max_parallel > gpu_count, предупреждаем о проблеме с логированием мощности
        if self.max_parallel > self.gpu_count:
            logger.warning(f"⚠️  max_parallel ({self.max_parallel}) больше количества GPU ({self.gpu_count})")
            logger.warning(f"💡 Несколько экспериментов могут использовать одну GPU одновременно")
            logger.warning(f"⚠️  ВНИМАНИЕ: Мощность GPU будет логироваться как общая для всех процессов на GPU!")
            logger.warning(f"💡 Рекомендуется установить --max-parallel {self.gpu_count} для корректного логирования мощности")
        elif self.max_parallel < self.gpu_count:
            logger.warning(f"⚠️  max_parallel ({self.max_parallel}) меньше количества GPU ({self.gpu_count})")
            logger.warning(f"💡 Не все GPU будут использоваться одновременно. Рекомендуется установить --max-parallel {self.gpu_count}")
        else:
            logger.info(f"✅ Настроено для использования всех {self.gpu_count} GPU параллельно")
            logger.info(f"✅ Гарантируется эксклюзивное использование GPU (одна GPU = один эксперимент)")
            logger.info(f"✅ Мощность GPU будет логироваться корректно для каждого эксперимента")
        
        # Очередь задач
        self.task_queue = Queue()
        self.running_tasks: Dict[int, ExperimentTask] = {}
        self.completed_tasks: List[ExperimentTask] = []
        self.failed_tasks: List[ExperimentTask] = []
        
        # Блокировка для потокобезопасности
        self.lock = threading.Lock()
        
        # Оценка памяти для моделей (в GB)
        self.model_memory_estimates = self._load_model_memory_estimates()
        
        # ClearML задача для логирования прогресса
        self.clearml_task = None
        self.clearml_logger = None
        if self.use_clearml:
            self._setup_clearml()
    
    def _setup_clearml(self):
        """Настроить ClearML для логирования прогресса."""
        try:
            from clearml import Task, Logger
            from dotenv import load_dotenv
            import os
            
            load_dotenv()
            
            self.clearml_task = Task.create(
                project_name=self.clearml_project or "slm-experiments",
                task_name=f"batch_experiments_{self.experiment_mode}",
                task_type=Task.TaskTypes.custom
            )
            
            # Логируем конфигурацию пакетного запуска
            connect_config = {
                "models": self.models,
                "datasets": self.datasets,
                "experiment_mode": self.experiment_mode,
                "max_parallel": self.max_parallel,
                "gpu_count": self.gpu_count,
                "retry_count": self.max_retries
            }
            if self.clearml_project:
                connect_config["clearml_project"] = self.clearml_project
            if self.output_dir:
                connect_config["output_dir"] = self.output_dir
            self.clearml_task.connect(connect_config)
            
            self.clearml_logger = Logger.current_logger()
            logger.info("✅ ClearML задача создана для логирования прогресса")
        except Exception as e:
            logger.warning(f"⚠️  Не удалось настроить ClearML: {e}")
            self.use_clearml = False
    
    def _get_s3_bucket_for_experiment_mode(self) -> str:
        """Возвращает bucket MinIO = имя конфига experiment_mode.
        
        Подчёркивания заменяются на дефисы (S3/MinIO не допускают _ в именах bucket).
        """
        return self.experiment_mode.replace("_", "-")
    
    def _load_model_memory_estimates(self) -> Dict[str, float]:
        """Загрузить оценки памяти для моделей из конфигов."""
        estimates = {}
        configs_dir = Path("configs/model")
        
        for model_file in configs_dir.glob("*.yaml"):
            try:
                with open(model_file, 'r') as f:
                    config = yaml.safe_load(f)
                    model_name = config.get('name', model_file.stem)
                    model_size = config.get('model_size', '0B')
                    
                    # Парсим размер модели (например, "0.6B", "1.7B", "4B")
                    size_str = model_size.replace('B', '').strip()
                    try:
                        size_value = float(size_str)
                        # Примерная оценка: 1B параметров ≈ 2GB памяти (float16)
                        # Добавляем запас для активаций и буферов
                        estimated_gb = size_value * 2.5  # Коэффициент с запасом
                        estimates[model_name] = estimated_gb
                        logger.info(f"📊 Модель {model_name}: оценка памяти ~{estimated_gb:.1f}GB")
                    except ValueError:
                        # Если не удалось распарсить, используем значения по умолчанию
                        if '0.6' in size_str or '135m' in model_name.lower():
                            estimates[model_name] = 2.0
                        elif '1.7' in size_str or '360m' in model_name.lower():
                            estimates[model_name] = 4.0
                        elif '4' in size_str:
                            estimates[model_name] = 9.0
                        else:
                            estimates[model_name] = 3.0
            except Exception as e:
                logger.warning(f"⚠️  Не удалось загрузить конфиг для {model_file}: {e}")
        
        return estimates
    
    def generate_tasks(self):
        """Генерировать все задачи экспериментов."""
        tasks = []
        
        for model in self.models:
            for dataset in self.datasets:
                estimated_memory = self.model_memory_estimates.get(model, 3.0)
                task_id = f"{model}_{dataset}_{self.experiment_mode}"
                task = ExperimentTask(
                    model=model,
                    dataset=dataset,
                    experiment_mode=self.experiment_mode,
                    estimated_memory_gb=estimated_memory,
                    max_retries=self.max_retries,
                    task_id=task_id
                )
                tasks.append(task)
                self.task_queue.put(task)
        
        # Сортируем задачи по размеру памяти (маленькие первыми для быстрого прогресса)
        tasks_sorted = sorted(tasks, key=lambda t: t.estimated_memory_gb)
        logger.info(f"📋 Сгенерировано {len(tasks)} задач экспериментов")
        logger.info(f"📊 Распределение по памяти:")
        for task in tasks_sorted:
            logger.info(f"   {task.model} × {task.dataset}: ~{task.estimated_memory_gb:.1f}GB")
        
        return len(tasks)
    
    def run_experiment(self, task: ExperimentTask, gpu_id: int) -> bool:
        """
        Запустить один эксперимент в Docker контейнере.
        
        Args:
            task: Задача эксперимента
            gpu_id: ID GPU для использования
        
        Returns:
            True если успешно, False иначе
        """
        import os
        
        # Определяем пути и настройки Docker
        workspace_dir = os.getcwd()
        cache_dir = os.environ.get("DOCKER_MODELS_CACHE", "/storage/docker-models")
        image_name = "slm-experiments:latest"
        network_name = "clearml_backend"
        
        # Проверяем, что Docker образ существует
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image_name],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError:
            logger.error(f"❌ Docker образ {image_name} не найден!")
            logger.error(f"💡 Соберите образ командой: ./build_docker_image.sh")
            return False
        
        # Проверяем, что Docker сеть существует
        try:
            subprocess.run(
                ["docker", "network", "inspect", network_name],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError:
            logger.error(f"❌ Docker сеть {network_name} не найдена!")
            return False
        
        # Bucket MinIO = имя конфига experiment_mode
        s3_bucket = self._get_s3_bucket_for_experiment_mode()
        logger.info(f"📦 MinIO bucket: {s3_bucket} (совпадает с experiment_mode)")
        
        # Формируем Hydra overrides для команды
        hydra_overrides = [
            f"model={task.model}",
            f"dataset={task.dataset}",
            f"experiment_mode={task.experiment_mode}"
        ]
        if self.output_dir:
            # Оборачиваем в кавычки, чтобы bash не раскрывал ${experiment.name}
            hydra_overrides.append(f"experiment.output_dir='{self.output_dir}/${{experiment.name}}'")
        if self.clearml_project:
            hydra_overrides.append(f"experiment.clearml_project={self.clearml_project}")

        run_cmd = "python run_experiment_simple.py " + " ".join(hydra_overrides)

        # Формируем команду для запуска эксперимента через Docker
        # Используем --gpus device=N для выбора конкретной GPU
        # Все настройки из run_experiment_fast.sh, но с выбором конкретной GPU
        docker_cmd = [
            "docker", "run", "--rm",
            "--network", network_name,
            "--gpus", f"device={gpu_id}",  # Выбираем конкретную GPU
            "-v", f"{workspace_dir}:/workspace",
            "-v", f"{workspace_dir}/clearml.conf.docker:/root/.clearml.conf:ro",
            "-v", f"{workspace_dir}/.env:/workspace/.env:ro",
            "-v", f"{cache_dir}/huggingface:/root/.cache/huggingface",
            "-v", f"{cache_dir}/datasets:/root/.cache/datasets",
            "-w", "/workspace",
            "-e", "PYTHONPATH=/workspace",
            "-e", "TRANSFORMERS_CACHE=/root/.cache/huggingface",
            "-e", "HF_HOME=/root/.cache/huggingface",
            "-e", "CLEARML_S3_ENDPOINT=http://minio:9000",
            "-e", f"CLEARML_S3_BUCKET={s3_bucket}",
            "-e", "CLEARML_S3_ACCESS_KEY=minioadmin",
            "-e", "CLEARML_S3_SECRET_KEY=minioadmin",
            "-e", "CLEARML_S3_REGION=us-east-1",
            image_name,
            "bash", "-c",
            run_cmd
        ]
        
        logger.info(f"🚀 Запуск в Docker: {task.model} × {task.dataset} на GPU {gpu_id}")
        logger.debug(f"   Полная команда: {' '.join(docker_cmd)}")
        
        try:
            # Запускаем эксперимент
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Логируем вывод в реальном времени
            def log_output(pipe, prefix):
                for line in iter(pipe.readline, ''):
                    if line:
                        logger.info(f"[GPU{gpu_id}][{task.model}×{task.dataset}] {prefix}: {line.rstrip()}")
            
            stdout_thread = threading.Thread(target=log_output, args=(process.stdout, "STDOUT"))
            stderr_thread = threading.Thread(target=log_output, args=(process.stderr, "STDERR"))
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()
            
            # Ждем завершения
            return_code = process.wait()
            
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            
            if return_code == 0:
                logger.info(f"✅ Завершен: {task.model} × {task.dataset} на GPU {gpu_id}")
                return True
            else:
                logger.error(f"❌ Ошибка (код {return_code}): {task.model} × {task.dataset} на GPU {gpu_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Исключение при запуске эксперимента: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def worker_thread(self, worker_id: int):
        """Поток-воркер для выполнения экспериментов."""
        logger.info(f"🔄 Воркер {worker_id} запущен")
        
        while True:
            try:
                # Получаем задачу из очереди
                task = self.task_queue.get(timeout=1)
                
                if task is None:  # Сигнал завершения
                    break
                
                # Сразу отмечаем задачу как обрабатываемую, чтобы основной цикл не завершился преждевременно
                with self.lock:
                    task.status = "waiting_for_gpu"
                    self.running_tasks[worker_id] = task
                
                # Ищем доступную GPU
                # Будем периодически проверять освобождение GPU без ограничения по времени
                # Это позволяет запускать эксперименты по мере освобождения GPU
                gpu_id = None
                wait_start = time.time()
                last_log_time = wait_start
                check_interval = 10  # Проверяем каждые 10 секунд
                
                logger.info(f"🔍 Воркер {worker_id} ищет свободную GPU для {task.model} × {task.dataset} (требуется ~{task.estimated_memory_gb:.1f}GB)")
                
                while gpu_id is None:
                    gpu_id = self.gpu_monitor.find_available_gpu(
                        task.estimated_memory_gb,
                        reserved_memory_gb=2.0,
                        task_id=task.task_id
                    )
                    
                    if gpu_id is None:
                        elapsed = time.time() - wait_start
                        
                        # Логируем прогресс каждые 60 секунд
                        if time.time() - last_log_time >= 60:
                            logger.info(
                                f"⏳ Воркер {worker_id} ждет свободную GPU для {task.model} × {task.dataset} "
                                f"(прошло {elapsed/60:.1f} мин, требуется ~{task.estimated_memory_gb:.1f}GB). "
                                f"Периодически проверяю освобождение GPU..."
                            )
                            last_log_time = time.time()
                        
                        # Ждем и проверяем снова
                        time.sleep(check_interval)
                
                if gpu_id is not None:
                    # Сохраняем информацию о GPU в задаче
                    task.gpu_id = gpu_id
                    
                    # Обновляем статус задачи на "running"
                    with self.lock:
                        task.status = "running"
                        # Задача уже в running_tasks, просто обновляем статус
                        self.running_tasks[worker_id] = task
                    
                    logger.info(f"🎯 Воркер {worker_id}: {task.model} × {task.dataset} назначен на GPU {gpu_id}")
                    
                    success = False
                    try:
                        # Запускаем эксперимент
                        success = self.run_experiment(task, gpu_id)
                    except Exception as exp_error:
                        logger.error(f"❌ Исключение при выполнении эксперимента {task.model} × {task.dataset}: {exp_error}")
                        import traceback
                        logger.error(traceback.format_exc())
                        success = False
                    finally:
                        # Всегда освобождаем GPU и удаляем задачу из running_tasks
                        try:
                            self.gpu_monitor.release_gpu(gpu_id, task.task_id)
                        except Exception as release_error:
                            logger.error(f"❌ Ошибка при освобождении GPU {gpu_id}: {release_error}")
                        
                        # Обновляем статус
                        with self.lock:
                            if success:
                                task.status = "completed"
                                self.completed_tasks.append(task)
                            else:
                                # Повторяем при ошибке
                                if task.retry_count < task.max_retries:
                                    task.retry_count += 1
                                    task.status = "pending"
                                    task.gpu_id = None  # Сбрасываем GPU для повторной попытки
                                    logger.info(f"🔄 Повтор {task.retry_count}/{task.max_retries}: {task.model} × {task.dataset}")
                                    self.task_queue.put(task)  # Возвращаем в очередь
                                else:
                                    task.status = "failed"
                                    self.failed_tasks.append(task)
                                    logger.error(f"❌ Превышено количество попыток: {task.model} × {task.dataset}")
                            
                            # Всегда удаляем задачу из running_tasks
                            if worker_id in self.running_tasks:
                                del self.running_tasks[worker_id]
                
                self.task_queue.task_done()
                
            except QueueEmpty:
                # Это нормально - просто продолжаем ждать
                continue
            except Exception as e:
                logger.error(f"❌ Ошибка в воркере {worker_id}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Убеждаемся, что задача удалена из running_tasks при ошибке
                with self.lock:
                    if worker_id in self.running_tasks:
                        task = self.running_tasks[worker_id]
                        logger.error(f"❌ Задача {task.model} × {task.dataset} была прервана из-за ошибки в воркере")
                        # Освобождаем GPU если она была назначена
                        if task.gpu_id is not None:
                            try:
                                self.gpu_monitor.release_gpu(task.gpu_id, task.task_id)
                            except Exception as release_error:
                                logger.error(f"❌ Ошибка при освобождении GPU {task.gpu_id}: {release_error}")
                        # Помечаем задачу как failed
                        task.status = "failed"
                        task.failure_reason = f"Ошибка в воркере: {str(e)}"
                        self.failed_tasks.append(task)
                        del self.running_tasks[worker_id]
                    # Если задача была получена, но еще не обработана, помечаем task_done
                    try:
                        self.task_queue.task_done()
                    except ValueError:
                        # task_done уже был вызван или задача не была получена
                        pass
    
    def run(self):
        """Запустить все эксперименты."""
        total_tasks = self.generate_tasks()
        
        if total_tasks == 0:
            logger.error("❌ Нет задач для выполнения")
            return
        
        logger.info(f"🚀 Запуск {total_tasks} экспериментов")
        logger.info(f"📊 Максимальное параллельное выполнение: {self.max_parallel}")
        logger.info(f"🎯 Доступно GPU: {self.gpu_count}")
        logger.info(f"💡 Все {self.gpu_count} GPU будут использоваться параллельно при наличии задач")
        
        # Запускаем воркеры
        workers = []
        for i in range(self.max_parallel):
            worker = threading.Thread(target=self.worker_thread, args=(i,))
            worker.daemon = True
            worker.start()
            workers.append(worker)
        
        # Мониторинг прогресса
        last_report_time = time.time()
        while not self.task_queue.empty() or self.running_tasks:
            # Логируем состояние цикла для отладки
            queue_size = self.task_queue.qsize()
            running_count = len(self.running_tasks)
            if queue_size > 0 or running_count > 0:
                logger.debug(f"🔍 Состояние: очередь={queue_size}, выполняющихся={running_count}")
            time.sleep(10)
            
            # Периодический отчет о прогрессе
            if time.time() - last_report_time > 60:  # Каждую минуту
                completed = len(self.completed_tasks)
                failed = len(self.failed_tasks)
                running = len(self.running_tasks)
                remaining = self.task_queue.qsize()
                total = completed + failed + running + remaining
                progress_pct = (completed / total * 100) if total > 0 else 0
                
                logger.info(f"📊 Прогресс: завершено {completed}/{total} ({progress_pct:.1f}%), "
                          f"выполняется {running}, осталось {remaining}, ошибок {failed}")
                
                # Логируем в ClearML
                if self.clearml_logger:
                    self.clearml_logger.report_scalar(
                        title="Batch Progress",
                        series="completed",
                        value=completed,
                        iteration=int(time.time())
                    )
                    self.clearml_logger.report_scalar(
                        title="Batch Progress",
                        series="failed",
                        value=failed,
                        iteration=int(time.time())
                    )
                    self.clearml_logger.report_scalar(
                        title="Batch Progress",
                        series="running",
                        value=running,
                        iteration=int(time.time())
                    )
                    self.clearml_logger.report_scalar(
                        title="Batch Progress",
                        series="remaining",
                        value=remaining,
                        iteration=int(time.time())
                    )
                    self.clearml_logger.report_scalar(
                        title="Batch Progress",
                        series="progress_percent",
                        value=progress_pct,
                        iteration=int(time.time())
                    )
                
                # Показываем состояние GPU
                free_memories = self.gpu_monitor.get_free_memory_per_gpu()
                total_memories = self.gpu_monitor.get_total_memory_per_gpu()
                
                logger.info(f"💻 Состояние GPU:")
                for gpu_idx, (free_mem, total_mem) in enumerate(zip(free_memories, total_memories)):
                    used_mem = total_mem - free_mem
                    usage_pct = (used_mem / total_mem * 100) if total_mem > 0 else 0
                    
                    # Проверяем, какая задача использует эту GPU
                    running_task_info = ""
                    for worker_id, running_task in self.running_tasks.items():
                        if running_task.gpu_id == gpu_idx:
                            running_task_info = f" [{running_task.model}×{running_task.dataset}]"
                            break
                    
                    status = "🟢 свободна" if gpu_idx not in self.gpu_monitor.gpu_reservations else "🔴 занята"
                    logger.info(f"   GPU {gpu_idx}: {status} | использовано {used_mem:.1f}GB/{total_mem:.1f}GB ({usage_pct:.1f}%){running_task_info}")
                    
                    if self.clearml_logger:
                        self.clearml_logger.report_scalar(
                            title="GPU Memory",
                            series=f"gpu_{gpu_idx}_free_gb",
                            value=free_mem,
                            iteration=int(time.time())
                        )
                        self.clearml_logger.report_scalar(
                            title="GPU Memory",
                            series=f"gpu_{gpu_idx}_used_gb",
                            value=used_mem,
                            iteration=int(time.time())
                        )
                        self.clearml_logger.report_scalar(
                            title="GPU Memory",
                            series=f"gpu_{gpu_idx}_usage_pct",
                            value=usage_pct,
                            iteration=int(time.time())
                        )
                
                last_report_time = time.time()
        
        # Дополнительная проверка перед завершением
        logger.info(f"🔍 Проверка перед завершением: очередь={self.task_queue.qsize()}, выполняющихся={len(self.running_tasks)}")
        if self.running_tasks:
            logger.warning(f"⚠️  Обнаружены выполняющиеся задачи перед завершением: {list(self.running_tasks.keys())}")
            # Ждем еще немного, чтобы дать воркерам время завершиться
            time.sleep(5)
        
        # Ждем завершения всех воркеров
        logger.info("🛑 Отправка сигналов завершения воркерам...")
        for _ in workers:
            self.task_queue.put(None)  # Сигнал завершения
        
        for worker in workers:
            worker.join(timeout=10)
        
        # Финальный отчет
        logger.info("=" * 80)
        logger.info("📊 ФИНАЛЬНЫЙ ОТЧЕТ")
        logger.info("=" * 80)
        logger.info(f"✅ Успешно завершено: {len(self.completed_tasks)}")
        logger.info(f"❌ Ошибок: {len(self.failed_tasks)}")
        logger.info(f"📋 Всего задач: {total_tasks}")
        
        # Логируем финальный отчет в ClearML
        if self.clearml_logger:
            self.clearml_logger.report_text("=" * 80)
            self.clearml_logger.report_text("📊 ФИНАЛЬНЫЙ ОТЧЕТ ПАКЕТНЫХ ЭКСПЕРИМЕНТОВ")
            self.clearml_logger.report_text("=" * 80)
            self.clearml_logger.report_text(f"✅ Успешно завершено: {len(self.completed_tasks)}")
            self.clearml_logger.report_text(f"❌ Ошибок: {len(self.failed_tasks)}")
            self.clearml_logger.report_text(f"📋 Всего задач: {total_tasks}")
            
            if self.completed_tasks:
                self.clearml_logger.report_text("\n✅ Успешные эксперименты:")
                for task in self.completed_tasks:
                    self.clearml_logger.report_text(f"   {task.model} × {task.dataset}")
        
        if self.failed_tasks:
            logger.info("\n❌ Неудачные эксперименты:")
            for task in self.failed_tasks:
                reason = f" ({task.failure_reason})" if task.failure_reason else ""
                logger.info(f"   {task.model} × {task.dataset} (попыток: {task.retry_count}){reason}")
            
            if self.clearml_logger:
                self.clearml_logger.report_text("\n❌ Неудачные эксперименты:")
                for task in self.failed_tasks:
                    reason = f" ({task.failure_reason})" if task.failure_reason else ""
                    self.clearml_logger.report_text(f"   {task.model} × {task.dataset} (попыток: {task.retry_count}){reason}")
        
        logger.info("=" * 80)
        
        # Закрываем ClearML задачу
        if self.clearml_task:
            self.clearml_task.close()


def main():
    """Главная функция."""
    import argparse
    from dotenv import load_dotenv

    load_dotenv()  # Загружаем .env (DOCKER_MODELS_CACHE и др.)

    parser = argparse.ArgumentParser(description='Запуск пакетных экспериментов')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Список моделей (по умолчанию все из configs/model/)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Список датасетов (по умолчанию local_nq, local_simple_qa)')
    parser.add_argument('--experiment-mode', default='no_context',
                        help='Режим эксперимента (по умолчанию: no_context)')
    parser.add_argument('--clearml-project', default=None,
                        help='Проект в ClearML для логирования (по умолчанию: slm-experiments)')
    parser.add_argument('--output-dir', default=None,
                        help='Базовая папка для сохранения результатов (по умолчанию: outputs)')
    parser.add_argument('--max-parallel', type=int, default=None,
                        help='Максимальное количество параллельных экспериментов')
    parser.add_argument('--retry-count', type=int, default=3,
                        help='Количество попыток при ошибке (по умолчанию: 3)')
    parser.add_argument('--no-clearml', action='store_true',
                        help='Отключить логирование в ClearML')
    
    args = parser.parse_args()
    
    # Загружаем список моделей из конфигов
    if args.models is None:
        models_dir = Path("configs/model")
        models = sorted([f.stem for f in models_dir.glob("*.yaml")])
        logger.info(f"📦 Автоопределение моделей: {models}")
        logger.info(f"   Всего моделей: {len(models)}")
    else:
        models = args.models
        logger.info(f"📦 Использованы указанные модели: {models}")
    
    # Загружаем список датасетов из конфигов
    if args.datasets is None:
        datasets_dir = Path("configs/dataset")
        datasets = sorted([f.stem for f in datasets_dir.glob("*.yaml")])
        logger.info(f"📊 Автоопределение датасетов: {datasets}")
        logger.info(f"   Всего датасетов: {len(datasets)}")
    else:
        datasets = args.datasets
        logger.info(f"📊 Использованы указанные датасеты: {datasets}")
    
    # oracle_long_context: simple_qa не имеет long_answer, исключаем из экспериментов
    if args.experiment_mode == "oracle_long_context":
        datasets_with_long = [d for d in datasets if "simple_qa" not in d.lower()]
        if datasets_with_long != datasets:
            logger.info(f"📊 oracle_long_context: исключён simple_qa (нет long_answer), датасеты: {datasets_with_long}")
            datasets = datasets_with_long
    
    # Показываем общее количество экспериментов
    total_experiments = len(models) * len(datasets)
    logger.info(f"🎯 Всего будет запущено экспериментов: {total_experiments} ({len(models)} моделей × {len(datasets)} датасетов)")
    
    # Создаем и запускаем менеджер экспериментов
    runner = BatchExperimentRunner(
        models=models,
        datasets=datasets,
        experiment_mode=args.experiment_mode,
        max_parallel=args.max_parallel,
        retry_count=args.retry_count,
        use_clearml=not args.no_clearml,
        clearml_project=args.clearml_project,
        output_dir=args.output_dir
    )
    
    runner.run()


if __name__ == "__main__":
    import os
    main()

