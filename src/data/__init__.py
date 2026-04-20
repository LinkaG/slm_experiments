"""Data module."""
from omegaconf import OmegaConf

from .base import BaseDataset, DatasetItem

def _config_dict(config):
    return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)


def get_dataset(config):
    """Factory function to get dataset based on config."""
    dataset_type = config.get("type", "dummy")

    if dataset_type == "qa_pairs_jsonl":
        from .qa_pairs_jsonl_dataset import QAPairsJsonlDataset
        return QAPairsJsonlDataset(_config_dict(config))

    elif dataset_type == "mirage":
        raise ValueError(
            "Тип датасета 'mirage' удалён. Укажите type: qa_pairs_jsonl и qa_pairs_path к qa_pairs.jsonl."
        )

    else:
        # Dummy dataset for testing
        from .base import BaseDataset, DatasetItem
        
        class DummyDataset(BaseDataset):
            """Dummy dataset for testing."""
            
            def __init__(self, config):
                self.config = config
            
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
                    "eval_size": 10,
                    "avg_context_length": 100
                }
        
        return DummyDataset(config)

__all__ = ['BaseDataset', 'DatasetItem', 'get_dataset']
