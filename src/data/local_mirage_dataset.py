from typing import Dict, Any, Iterator
from .local_dataset import LocalDataset, DatasetItem

class LocalMirageDataset(LocalDataset):
    """Local MIRAGE dataset implementation."""
    
    def _get_data_iterator(self, data: list) -> Iterator[DatasetItem]:
        """Convert MIRAGE format to DatasetItem iterator."""
        for item in data:
            yield DatasetItem(
                question=item['question'],
                answer=item.get('answer'),  # may be None for test set
                context=item.get('context'),  # may be None if document wasn't downloaded
                metadata=item.get('metadata', {})  # Все метаданные уже в metadata
            )

