import boto3
import json
from typing import Dict, Any, Iterator, Optional
import logging
from pathlib import Path
import tempfile

from .base import BaseDataset, DatasetItem

class S3Dataset(BaseDataset):
    """Base class for datasets stored in S3."""
    
    def __init__(self, dataset_config: Dict[str, Any]):
        """Initialize dataset with configuration.
        
        Expected config parameters:
        - aws_access_key_id: AWS access key
        - aws_secret_access_key: AWS secret key
        - region_name: AWS region
        - bucket: S3 bucket name
        - train_key: S3 key for training data
        - eval_key: S3 key for evaluation data
        - cache_dir: Local directory for caching (optional)
        """
        self.config = dataset_config
        self.logger = logging.getLogger(__name__)
        
        # Setup S3 client
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=dataset_config['aws_access_key_id'],
            aws_secret_access_key=dataset_config['aws_secret_access_key'],
            region_name=dataset_config['region_name']
        )
        
        # Setup cache
        self.cache_dir = Path(dataset_config.get('cache_dir', tempfile.gettempdir()))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.train_data = None
        self.eval_data = None
        
        # Load data
        self.load_from_s3(
            bucket=dataset_config['bucket'],
            train_key=dataset_config['train_key'],
            eval_key=dataset_config['eval_key']
        )
    
    def load_from_s3(self, bucket: str, train_key: str, eval_key: str):
        """Load dataset from S3 with caching."""
        self.train_data = self._load_split(bucket, train_key, "train")
        self.eval_data = self._load_split(bucket, eval_key, "eval")
    
    def _load_split(self, bucket: str, key: str, split: str) -> list:
        """Load and cache a data split."""
        cache_file = self.cache_dir / f"{bucket}_{key.replace('/', '_')}_{split}.json"
        
        # Try to load from cache first
        if cache_file.exists():
            self.logger.info(f"Loading {split} data from cache: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Download from S3
        self.logger.info(f"Downloading {split} data from S3: {bucket}/{key}")
        try:
            response = self.s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            
            # Cache the data
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data from S3: {e}")
            raise
    
    def get_train_data(self) -> Iterator[DatasetItem]:
        """Get training data iterator."""
        if self.train_data is None:
            raise ValueError("Dataset not loaded. Call load_from_s3 first.")
        return self._get_data_iterator(self.train_data)
    
    def get_eval_data(self) -> Iterator[DatasetItem]:
        """Get evaluation data iterator."""
        if self.eval_data is None:
            raise ValueError("Dataset not loaded. Call load_from_s3 first.")
        return self._get_data_iterator(self.eval_data)
    
    def _get_data_iterator(self, data: list) -> Iterator[DatasetItem]:
        """Convert raw data to DatasetItem iterator."""
        raise NotImplementedError("Implement in derived class")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        return {
            "train_size": len(self.train_data) if self.train_data else 0,
            "eval_size": len(self.eval_data) if self.eval_data else 0
        }
