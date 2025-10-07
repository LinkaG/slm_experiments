import json
from typing import Dict, Any, Iterator, Optional
import logging
from pathlib import Path

from .base import BaseDataset, DatasetItem

class LocalDataset(BaseDataset):
    """Base class for datasets stored locally."""
    
    def __init__(self, dataset_config: Dict[str, Any]):
        """Initialize dataset with configuration.
        
        Expected config parameters:
        - train_path: Path to training data file
        - eval_path: Path to evaluation data file
        - cache_dir: Local directory for caching (optional)
        """
        self.config = dataset_config
        self.logger = logging.getLogger(__name__)
        
        # Setup cache
        self.cache_dir = Path(dataset_config.get('cache_dir', '.cache/datasets'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.train_data = None
        self.eval_data = None
        
        # Load data
        self.load_from_local(
            train_path=dataset_config['train_path'],
            eval_path=dataset_config['eval_path']
        )
    
    def load_from_local(self, train_path: str, eval_path: str):
        """Load dataset from local files with caching."""
        self.train_data = self._load_split(train_path, "train")
        self.eval_data = self._load_split(eval_path, "eval")
    
    def _load_split(self, file_path: str, split: str) -> list:
        """Load and cache a data split."""
        file_path = Path(file_path)
        cache_file = self.cache_dir / f"{file_path.stem}_{split}.json"
        
        # Try to load from cache first
        if cache_file.exists():
            self.logger.info(f"Loading {split} data from cache: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Load from local file
        self.logger.info(f"Loading {split} data from file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Обрабатываем разные форматы JSON
            if isinstance(data, dict):
                # Новый формат с train/eval ключами
                if 'train' in data and 'eval' in data:
                    split_data = data[split]
                else:
                    # Старый формат - весь файл это массив
                    split_data = data
            else:
                # Старый формат - весь файл это массив
                split_data = data
            
            # Cache the data
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            return split_data
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            raise
    
    def get_train_data(self) -> Iterator[DatasetItem]:
        """Get training data iterator."""
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load_from_local first.")
        return self._get_data_iterator(self.train_data)
    
    def get_eval_data(self) -> Iterator[DatasetItem]:
        """Get evaluation data iterator."""
        if self.eval_data is None:
            raise ValueError("Dataset not loaded. Call load_from_local first.")
        return self._get_data_iterator(self.eval_data)
    
    def _get_data_iterator(self, data: list) -> Iterator[DatasetItem]:
        """Convert raw data to DatasetItem iterator."""
        raise NotImplementedError("Implement in derived class")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        stats = {
            'train_size': len(self.train_data) if self.train_data else 0,
            'eval_size': len(self.eval_data) if self.eval_data else 0,
        }
        return stats
