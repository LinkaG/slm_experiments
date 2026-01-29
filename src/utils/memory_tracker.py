import psutil
import torch
import gc
import subprocess
from typing import Dict, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import json

@dataclass
class MemoryStats:
    """Container for memory statistics."""
    cpu_ram_used: float  # в МБ
    gpu_ram_used: Optional[float]  # в МБ
    gpu_ram_peak: Optional[float]  # пиковое использование GPU RAM в МБ
    reserved_gpu_ram: Optional[float]  # зарезервированная GPU память
    gpu_power_draw: Optional[float]  # текущее потребление мощности GPU в Вт
    gpu_power_limit: Optional[float]  # лимит мощности GPU в Вт
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "cpu_ram_used_mb": self.cpu_ram_used,
            "gpu_ram_used_mb": self.gpu_ram_used if self.gpu_ram_used is not None else 0,
            "gpu_ram_peak_mb": self.gpu_ram_peak if self.gpu_ram_peak is not None else 0,
            "reserved_gpu_ram_mb": self.reserved_gpu_ram if self.reserved_gpu_ram is not None else 0,
            "gpu_power_draw_w": self.gpu_power_draw if self.gpu_power_draw is not None else 0,
            "gpu_power_limit_w": self.gpu_power_limit if self.gpu_power_limit is not None else 0
        }

class MemoryTracker:
    """Tracks CPU and GPU memory usage."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        self.memory_log = []
        self.peak_stats = None
        
    def get_current_memory_usage(self) -> MemoryStats:
        """
        Get current memory usage statistics.
        
        Важно: 
        - GPU память отслеживается только для текущего процесса Python через torch.cuda.memory_allocated().
          Если несколько экспериментов используют одну GPU параллельно, каждый эксперимент будет
          логировать только свою память, а не общую память GPU. Это правильное поведение.
        
        - GPU мощность отслеживается через nvidia-smi и показывает ОБЩЕЕ потребление всей GPU,
          а не конкретного процесса. Если несколько экспериментов используют одну GPU параллельно,
          каждый будет видеть общее потребление GPU, а не свое индивидуальное.
          Это ограничение nvidia-smi - он не может разделить мощность по процессам.
          
        Рекомендация: Используйте отдельную GPU для каждого эксперимента (по умолчанию так и есть),
        чтобы мощность логировалась корректно для каждого эксперимента.
        """
        # CPU Memory - отслеживается для текущего процесса
        process = psutil.Process()
        cpu_ram = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        
        # GPU Memory - отслеживается только для текущего процесса через PyTorch
        # torch.cuda.memory_allocated() показывает память, выделенную в текущем процессе
        # Это правильно, даже если несколько процессов используют одну GPU
        gpu_ram_used = None
        gpu_ram_peak = None
        reserved_gpu_ram = None
        
        # GPU Memory и Power
        gpu_power_draw = None
        gpu_power_limit = None
        
        if torch.cuda.is_available():
            # Память, выделенная через PyTorch в текущем процессе
            gpu_ram_used = torch.cuda.memory_allocated() / (1024 * 1024)
            # Пиковое использование памяти в текущем процессе
            gpu_ram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            # Зарезервированная память в текущем процессе
            reserved_gpu_ram = torch.cuda.memory_reserved() / (1024 * 1024)
            
            # Получаем информацию о мощности GPU через nvidia-smi
            # ВАЖНО: nvidia-smi показывает общее потребление мощности всей GPU, а не конкретного процесса.
            # Если несколько экспериментов используют одну GPU параллельно, каждый будет видеть
            # общее потребление GPU, а не свое индивидуальное. Это ограничение nvidia-smi.
            # 
            # В текущей реализации каждый эксперимент должен использовать отдельную GPU
            # (через --gpus device=N в Docker), поэтому мощность будет корректной для каждого эксперимента.
            # Но если установить --max-parallel больше количества GPU, несколько экспериментов
            # могут использовать одну GPU, и тогда мощность будет общей для всех.
            try:
                gpu_device_id = torch.cuda.current_device()
                # Запрашиваем мощность для конкретного GPU
                # Это показывает общее потребление всей GPU, а не только текущего процесса
                result = subprocess.run(
                    ['nvidia-smi', 
                     '--query-gpu=index,power.draw,power.limit',
                     '--format=csv,noheader,nounits',
                     f'--id={gpu_device_id}'],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True
                )
                
                # Парсим результат: "0, 125.50, 250.00"
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_idx = int(parts[0].strip())
                        if gpu_idx == gpu_device_id:
                            try:
                                gpu_power_draw = float(parts[1].strip())
                                gpu_power_limit = float(parts[2].strip())
                                
                                # Предупреждение, если мощность очень высокая (возможно, несколько процессов)
                                # Это эвристика - если мощность близка к лимиту, возможно несколько процессов
                                if gpu_power_limit > 0 and gpu_power_draw / gpu_power_limit > 0.9:
                                    self.logger.debug(
                                        f"⚠️  Высокое потребление мощности GPU ({gpu_power_draw:.1f}W/{gpu_power_limit:.1f}W). "
                                        f"Возможно, несколько процессов используют эту GPU одновременно."
                                    )
                            except (ValueError, IndexError):
                                pass
                            break
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
                # nvidia-smi недоступен или произошла ошибка - это не критично
                self.logger.debug(f"Не удалось получить информацию о мощности GPU: {e}")
        else:
            gpu_ram_used = None
            gpu_ram_peak = None
            reserved_gpu_ram = None
        
        stats = MemoryStats(
            cpu_ram_used=cpu_ram,
            gpu_ram_used=gpu_ram_used,
            gpu_ram_peak=gpu_ram_peak,
            reserved_gpu_ram=reserved_gpu_ram,
            gpu_power_draw=gpu_power_draw,
            gpu_power_limit=gpu_power_limit
        )
        
        # Update peak stats
        if self.peak_stats is None:
            self.peak_stats = stats
        else:
            self.peak_stats = MemoryStats(
                cpu_ram_used=max(self.peak_stats.cpu_ram_used, stats.cpu_ram_used),
                gpu_ram_used=max(self.peak_stats.gpu_ram_used or 0, stats.gpu_ram_used or 0),
                gpu_ram_peak=max(self.peak_stats.gpu_ram_peak or 0, stats.gpu_ram_peak or 0),
                reserved_gpu_ram=max(self.peak_stats.reserved_gpu_ram or 0, stats.reserved_gpu_ram or 0),
                gpu_power_draw=max(self.peak_stats.gpu_power_draw or 0, stats.gpu_power_draw or 0) if stats.gpu_power_draw else self.peak_stats.gpu_power_draw,
                gpu_power_limit=stats.gpu_power_limit if stats.gpu_power_limit else self.peak_stats.gpu_power_limit
            )
        
        return stats
    
    def log_memory(self, component: str, operation: str):
        """Log memory usage for specific component and operation."""
        stats = self.get_current_memory_usage()
        log_entry = {
            "component": component,
            "operation": operation,
            "timestamp": psutil.time.time(),
            **stats.to_dict()
        }
        self.memory_log.append(log_entry)
        
        # Log to ClearML
        from clearml import Logger
        logger = Logger.current_logger()
        if logger:
            for key, value in stats.to_dict().items():
                logger.report_scalar(
                    title=f"memory/{component}",
                    series=key,
                    value=value,
                    iteration=len(self.memory_log)
                )
    
    def clear_memory(self):
        """Clear unused memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def save_log(self):
        """Save memory log to file."""
        log_file = self.log_dir / "memory_usage.json"
        with open(log_file, "w") as f:
            json.dump({
                "detailed_log": self.memory_log,
                "peak_usage": self.peak_stats.to_dict() if self.peak_stats else None
            }, f, indent=2)
        
        self.logger.info(f"Memory usage log saved to {log_file}")
        
        # Log peak usage summary
        if self.peak_stats:
            self.logger.info("Peak memory usage:")
            for key, value in self.peak_stats.to_dict().items():
                self.logger.info(f"  {key}: {value:.2f} MB")
