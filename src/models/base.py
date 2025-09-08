from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseModel(ABC):
    """Base interface for all language models."""
    
    @abstractmethod
    def __init__(self, model_config: Dict[str, Any]):
        """Initialize model with configuration."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, context: List[str]) -> str:
        """Generate response given prompt and retrieved context."""
        pass
    
    @abstractmethod
    def get_model_size(self) -> str:
        """Return model size (e.g., '135M', '1.7B')."""
        pass
