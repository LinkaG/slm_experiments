"""Data module."""
from .base import BaseDataset, DatasetItem

def get_dataset(config):
    """Factory function to get dataset based on config."""
    dataset_type = config.get('type', 'dummy')
    use_local = config.get('use_local', False)
    
    if dataset_type == 'nq':
        if use_local:
            from .local_nq_dataset import LocalNaturalQuestionsDataset
            return LocalNaturalQuestionsDataset(dict(config))
        else:
            from .nq_dataset import NaturalQuestionsDataset
            return NaturalQuestionsDataset(dict(config))
    
    elif dataset_type == 'simple_qa':
        if use_local:
            from .local_simple_qa_dataset import LocalSimpleQADataset
            return LocalSimpleQADataset(dict(config))
        else:
            from .simple_qa_dataset import SimpleQADataset
            return SimpleQADataset(dict(config))
    
    else:
        # Dummy dataset for testing
        from .base import BaseDataset, DatasetItem
        
        class DummyDataset(BaseDataset):
            """Dummy dataset for testing."""
            
            def __init__(self, config):
                self.config = config
                
            def get_train_data(self):
                """Get training data."""
                return []
            
            def get_eval_data(self):
                """Get evaluation data."""
                # Возвращаем 10 тестовых примеров
                for i in range(10):
                    yield DatasetItem(
                        question=f"Test question {i}",
                        answer=f"Test answer {i}",
                        context=f"Test context {i}",
                        metadata={"id": i}
                    )
            
            def get_dataset_stats(self):
                """Get dataset statistics."""
                return {
                    "num_train": 0,
                    "num_eval": 10,
                    "avg_context_length": 100
                }
        
        return DummyDataset(config)

__all__ = ['BaseDataset', 'DatasetItem', 'get_dataset']
