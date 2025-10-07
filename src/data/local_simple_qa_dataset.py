from typing import Dict, Any, Iterator
from .local_dataset import LocalDataset, DatasetItem

class LocalSimpleQADataset(LocalDataset):
    """Local SimpleQA dataset implementation."""
    
    def _get_data_iterator(self, data: list) -> Iterator[DatasetItem]:
        """Convert SimpleQA format to DatasetItem iterator."""
        for item in data:
            # Обрабатываем контекст (может быть списком документов)
            context = item.get('context')
            if isinstance(context, list):
                # Если контекст - это список документов, объединяем их
                context = ' '.join(context)
            
            yield DatasetItem(
                question=item['question'],
                answer=item.get('answer'),  # may be None for test set
                context=context,  # SimpleQA имеет контекст (документы)
                metadata=item.get('metadata', {})
            )
