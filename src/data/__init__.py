from .base import BaseDataset, DatasetItem
from .nq_dataset import NaturalQuestionsDataset
from .simple_qa_dataset import SimpleQADataset
from .local_nq_dataset import LocalNaturalQuestionsDataset
from .local_simple_qa_dataset import LocalSimpleQADataset

def get_dataset(dataset_config):
    """Factory function to create dataset instances."""
    dataset_type = dataset_config.get('type', 'nq')
    use_local = dataset_config.get('use_local', False)
    
    if dataset_type == 'nq':
        if use_local:
            return LocalNaturalQuestionsDataset(dataset_config)
        else:
            return NaturalQuestionsDataset(dataset_config)
    elif dataset_type == 'simple_qa':
        if use_local:
            return LocalSimpleQADataset(dataset_config)
        else:
            return SimpleQADataset(dataset_config)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
