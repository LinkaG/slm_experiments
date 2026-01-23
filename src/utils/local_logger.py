"""
Модуль для локального сохранения результатов экспериментов без ClearML.
Сохраняет все данные в структурированном формате для последующей загрузки в ClearML.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class LocalLogger:
    """Локальный логгер для сохранения результатов без ClearML."""
    
    def __init__(self, output_dir: Path):
        """
        Инициализирует локальный логгер.
        
        Args:
            output_dir: Директория для сохранения результатов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Создаем поддиректории для разных типов данных
        self.metrics_dir = self.output_dir / "metrics"
        self.logs_dir = self.output_dir / "logs"
        self.artifacts_dir = self.output_dir / "artifacts"
        self.config_dir = self.output_dir / "config"
        
        for dir_path in [self.metrics_dir, self.logs_dir, self.artifacts_dir, self.config_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Хранилище данных
        self.text_logs = []
        self.scalar_metrics = {}  # {title: {series: [(iteration, value), ...]}}
        self.single_values = {}
        self.tables = []
        self.config = {}
        
    def report_text(self, text: str):
        """Сохраняет текстовый лог."""
        timestamp = datetime.now().isoformat()
        self.text_logs.append({
            "timestamp": timestamp,
            "text": text
        })
        # Также выводим в консоль
        self.logger.info(text)
    
    def report_scalar(self, title: str, series: str, value: float, iteration: int = 0):
        """Сохраняет скалярную метрику."""
        if title not in self.scalar_metrics:
            self.scalar_metrics[title] = {}
        if series not in self.scalar_metrics[title]:
            self.scalar_metrics[title][series] = []
        
        self.scalar_metrics[title][series].append({
            "iteration": iteration,
            "value": value,
            "timestamp": datetime.now().isoformat()
        })
    
    def report_single_value(self, name: str, value: Any):
        """Сохраняет одиночное значение."""
        self.single_values[name] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
    
    def report_table(self, title: str, series: str, table_plot, iteration: int = 0):
        """Сохраняет таблицу."""
        if HAS_PANDAS and isinstance(table_plot, pd.DataFrame):
            data = table_plot.to_dict('records')
            columns = list(table_plot.columns)
        else:
            # Fallback для случаев без pandas
            data = table_plot if isinstance(table_plot, list) else []
            columns = list(table_plot[0].keys()) if data else []
        
        self.tables.append({
            "title": title,
            "series": series,
            "iteration": iteration,
            "data": data,
            "columns": columns,
            "timestamp": datetime.now().isoformat()
        })
    
    def info(self, message: str):
        """Стандартный метод логирования."""
        self.report_text(message)
    
    def save_all(self, experiment_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Сохраняет все данные в файлы.
        
        Args:
            experiment_name: Название эксперимента
            config: Конфигурация эксперимента
        """
        if config:
            self.config = config
        
        # Сохраняем метаданные эксперимента
        metadata = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "format_version": "1.0"
        }
        
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Сохраняем конфигурацию
        if self.config:
            with open(self.config_dir / "experiment_config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        # Сохраняем текстовые логи
        with open(self.logs_dir / "text_logs.json", "w", encoding="utf-8") as f:
            json.dump(self.text_logs, f, indent=2, ensure_ascii=False)
        
        # Сохраняем скалярные метрики
        with open(self.metrics_dir / "scalar_metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.scalar_metrics, f, indent=2, ensure_ascii=False)
        
        # Сохраняем одиночные значения
        with open(self.metrics_dir / "single_values.json", "w", encoding="utf-8") as f:
            json.dump(self.single_values, f, indent=2, ensure_ascii=False)
        
        # Сохраняем таблицы
        with open(self.metrics_dir / "tables.json", "w", encoding="utf-8") as f:
            json.dump(self.tables, f, indent=2, ensure_ascii=False)
        
        # Сохраняем CSV файлы для скалярных метрик (удобно для анализа)
        if HAS_PANDAS:
            for title, series_dict in self.scalar_metrics.items():
                # Создаем DataFrame для каждой серии
                for series, values in series_dict.items():
                    df = pd.DataFrame(values)
                    safe_title = title.replace("/", "_").replace(" ", "_")
                    safe_series = series.replace("/", "_").replace(" ", "_")
                    csv_path = self.metrics_dir / f"{safe_title}_{safe_series}.csv"
                    df.to_csv(csv_path, index=False)
        
        self.logger.info(f"✅ Все данные сохранены в {self.output_dir}")
    
    def save_artifact(self, name: str, file_path: Path, metadata: Optional[Dict[str, Any]] = None):
        """
        Сохраняет артефакт (копирует файл в artifacts_dir).
        
        Args:
            name: Имя артефакта
            file_path: Путь к файлу
            metadata: Метаданные артефакта
        """
        if not file_path.exists():
            self.logger.warning(f"Файл не найден: {file_path}")
            return
        
        # Копируем файл
        import shutil
        dest_path = self.artifacts_dir / f"{name}_{file_path.name}"
        shutil.copy2(file_path, dest_path)
        
        # Сохраняем метаданные
        artifact_metadata = {
            "name": name,
            "original_path": str(file_path),
            "saved_path": str(dest_path),
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            artifact_metadata.update(metadata)
        
        artifacts_meta_file = self.artifacts_dir / "artifacts_metadata.json"
        artifacts_meta = []
        if artifacts_meta_file.exists():
            with open(artifacts_meta_file, "r", encoding="utf-8") as f:
                artifacts_meta = json.load(f)
        
        artifacts_meta.append(artifact_metadata)
        
        with open(artifacts_meta_file, "w", encoding="utf-8") as f:
            json.dump(artifacts_meta, f, indent=2, ensure_ascii=False)

