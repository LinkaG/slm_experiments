"""Wrapper for logger to support both ClearML and standard logging."""
import logging
from typing import Any


class LoggerWrapper:
    """Wrapper that provides unified interface for ClearML and standard Python logging."""
    
    def __init__(self, logger):
        """
        Initialize wrapper.
        
        Args:
            logger: Either ClearML Logger, LocalLogger, or standard Python logger
        """
        self.logger = logger
        # Проверяем тип логгера
        self._is_clearml = hasattr(logger, 'report_scalar') and hasattr(logger, 'report_text')
        self._is_local = hasattr(logger, 'save_all')  # LocalLogger имеет метод save_all
    
    def report_scalar(self, title: str, series: str, value: Any, iteration: int = 0):
        """Report a scalar value."""
        if self._is_clearml or self._is_local:
            self.logger.report_scalar(title, series, value, iteration)
        else:
            self.logger.info(f"📊 {title}/{series}: {value}")
    
    def report_text(self, text: str):
        """Report text."""
        if self._is_clearml or self._is_local:
            self.logger.report_text(text)
        else:
            self.logger.info(text)
    
    def report_table(self, title: str, series: str, table_plot, iteration: int = 0):
        """Report a table."""
        if self._is_clearml or self._is_local:
            self.logger.report_table(title, series, table_plot=table_plot, iteration=iteration)
        else:
            self.logger.info(f"📊 Table: {title}/{series}")
            self.logger.info(f"\n{table_plot}")
    
    def report_single_value(self, name: str, value):
        """Report a single value (appears in SCALARS without graph)."""
        if self._is_clearml or self._is_local:
            self.logger.report_single_value(name, value)
        else:
            self.logger.info(f"📈 {name}: {value}")
    
    def info(self, msg: str):
        """Log info message."""
        if hasattr(self.logger, 'info'):
            self.logger.info(msg)
        else:
            # ClearML logger doesn't have info, use report_text
            self.report_text(msg)
    
    def error(self, msg: str):
        """Log error message."""
        if hasattr(self.logger, 'error'):
            self.logger.error(msg)
        else:
            self.report_text(f"ERROR: {msg}")
    
    def warning(self, msg: str):
        """Log warning message."""
        if hasattr(self.logger, 'warning'):
            self.logger.warning(msg)
        else:
            self.report_text(f"WARNING: {msg}")

