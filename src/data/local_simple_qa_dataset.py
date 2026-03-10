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
            
            # Метаданные: merge base metadata + long_context/long_answer для oracle_long
            metadata = dict(item.get('metadata') or {})
            if item.get('long_context') is not None:
                metadata['long_context'] = item['long_context']
            if item.get('long_answer') is not None:
                metadata['long_answer'] = item['long_answer']
            
            yield DatasetItem(
                question=item['question'],
                answer=item.get('answer'),  # may be None for test set
                context=context,  # SimpleQA имеет контекст (документы)
                metadata=metadata
            )
