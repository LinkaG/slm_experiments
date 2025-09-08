from typing import Dict, Any, Iterator
from .s3_dataset import S3Dataset, DatasetItem

class NaturalQuestionsDataset(S3Dataset):
    """Natural Questions dataset implementation."""
    
    def _get_data_iterator(self, data: list) -> Iterator[DatasetItem]:
        """Convert NQ format to DatasetItem iterator."""
        for item in data:
            yield DatasetItem(
                question=item['question'],
                answer=item.get('answer'),  # may be None for test set
                context=item.get('context'),  # may be None if not provided
                metadata={
                    'id': item.get('id'),
                    'url': item.get('url'),
                    'title': item.get('title')
                }
            )
