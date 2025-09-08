from typing import Dict, Any, Iterator
from .s3_dataset import S3Dataset, DatasetItem

class SimpleQADataset(S3Dataset):
    """SimpleQA dataset implementation."""
    
    def _get_data_iterator(self, data: list) -> Iterator[DatasetItem]:
        """Convert SimpleQA format to DatasetItem iterator."""
        for item in data:
            yield DatasetItem(
                question=item['question'],
                answer=item.get('answer'),  # may be None for test set
                context=None,  # SimpleQA doesn't have context
                metadata={
                    'id': item.get('id'),
                    'category': item.get('category')
                }
            )
