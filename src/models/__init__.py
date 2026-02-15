"""Models module."""
from .base import BaseModel
from .huggingface_model import HuggingFaceModel


def get_model(config):
    """Factory function to get model based on config.
    
    Args:
        config: Model configuration
        
    Returns:
        Model instance
    """
    model_type = config.get('model_type', 'huggingface')
    
    if model_type in ['smollm2', 'qwen', 'huggingface', 'minicpm4']:
        # All these models use HuggingFace implementation
        return HuggingFaceModel(dict(config))
    else:
        # Fallback to dummy model for testing
        from .base import BaseModel
        
        class DummyModel(BaseModel):
            """Dummy model for testing."""
            
            def __init__(self, config):
                self.config = config
                
            def generate(self, prompt: str, context: list = None) -> str:
                """Generate answer."""
                return "Test answer"
            
            def get_model_size(self) -> int:
                """Get model size in bytes."""
                return 1024
        
        return DummyModel(config)


__all__ = ['BaseModel', 'HuggingFaceModel', 'get_model']

