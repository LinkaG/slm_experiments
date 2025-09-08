import psutil
import torch
import gc
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
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "cpu_ram_used_mb": self.cpu_ram_used,
            "gpu_ram_used_mb": self.gpu_ram_used if self.gpu_ram_used is not None else 0,
            "gpu_ram_peak_mb": self.gpu_ram_peak if self.gpu_ram_peak is not None else 0,
            "reserved_gpu_ram_mb": self.reserved_gpu_ram if self.reserved_gpu_ram is not None else 0
        }

class MemoryTracker:
    """Tracks CPU and GPU memory usage."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        self.memory_log = []
        self.peak_stats = None
        
    def get_current_memory_usage(self) -> MemoryStats:
        """Get current memory usage statistics."""
        # CPU Memory
        process = psutil.Process()
        cpu_ram = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        
        # GPU Memory
        gpu_ram_used = None
        gpu_ram_peak = None
        reserved_gpu_ram = None
        
        if torch.cuda.is_available():
            gpu_ram_used = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_ram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            reserved_gpu_ram = torch.cuda.memory_reserved() / (1024 * 1024)
        
        stats = MemoryStats(
            cpu_ram_used=cpu_ram,
            gpu_ram_used=gpu_ram_used,
            gpu_ram_peak=gpu_ram_peak,
            reserved_gpu_ram=reserved_gpu_ram
        )
        
        # Update peak stats
        if self.peak_stats is None:
            self.peak_stats = stats
        else:
            self.peak_stats = MemoryStats(
                cpu_ram_used=max(self.peak_stats.cpu_ram_used, stats.cpu_ram_used),
                gpu_ram_used=max(self.peak_stats.gpu_ram_used or 0, stats.gpu_ram_used or 0),
                gpu_ram_peak=max(self.peak_stats.gpu_ram_peak or 0, stats.gpu_ram_peak or 0),
                reserved_gpu_ram=max(self.peak_stats.reserved_gpu_ram or 0, stats.reserved_gpu_ram or 0)
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
