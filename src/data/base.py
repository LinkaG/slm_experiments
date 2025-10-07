from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Optional
from dataclasses import dataclass

@dataclass
class DatasetItem:
    """Container for dataset items."""
    question: str
    answer: Optional[str]
    context: Optional[str]
    metadata: Dict[str, Any]

class BaseDataset(ABC):
    """Base interface for all datasets."""
    
    @abstractmethod
    def __init__(self, dataset_config: Dict[str, Any]):
        """Initialize dataset with configuration."""
        pass
    
    def load_from_s3(self, bucket: str, key: str):
        """Load dataset from S3."""
        pass
    
    @abstractmethod
    def get_train_data(self) -> Iterator[DatasetItem]:
        """Get training data iterator."""
        pass
    
    @abstractmethod
    def get_eval_data(self) -> Iterator[DatasetItem]:
        """Get evaluation data iterator."""
        pass
    
    @abstractmethod
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Return dataset statistics."""
        pass
