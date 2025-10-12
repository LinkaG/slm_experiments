"""Retrievers module."""
from .base import BaseRetriever

def get_retriever(config):
    """Factory function to get retriever based on config."""
    # Возвращает None, если ретривер не используется
    return None

__all__ = ['BaseRetriever', 'get_retriever']
