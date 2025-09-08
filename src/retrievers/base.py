from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    context: str
    score: float
    metadata: Dict[str, Any]

class BaseRetriever(ABC):
    """Base interface for all retrievers."""
    
    @abstractmethod
    def __init__(self, retriever_config: Dict[str, Any]):
        """Initialize retriever with configuration."""
        pass
    
    @abstractmethod
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build search index from documents."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Retrieve relevant contexts for the query."""
        pass
    
    @abstractmethod
    def get_index_size(self) -> int:
        """Return index size in bytes."""
        pass
