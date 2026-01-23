from typing import List, Set, Optional
from transformers import AutoTokenizer
import re

class TokenRecallCalculator:
    """Calculator for token-based recall metric."""
    
    def __init__(self, tokenizer=None, tokenizer_name: Optional[str] = None):
        """
        Initialize with tokenizer.
        
        Args:
            tokenizer: Pre-initialized tokenizer (preferred - uses model's tokenizer)
            tokenizer_name: Tokenizer name to load (fallback if tokenizer not provided)
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            # Fallback to a general tokenizer (but this is not ideal)
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def get_tokens(self, text) -> Set[str]:
        """
        Convert text to a set of normalized tokens.
        
        Args:
            text: Input text (str, list, or other)
            
        Returns:
            Set of normalized token strings
        """
        # Handle list of answers
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        
        # Ensure text is string
        if not isinstance(text, str):
            text = str(text)
        
        if not text.strip():
            return set()
            
        # Tokenize
        try:
            tokens = self.tokenizer.tokenize(text)
        except Exception as e:
            # Fallback to simple whitespace tokenization if tokenizer fails
            tokens = text.lower().split()
        
        # Normalize tokens: remove subword markers and special characters
        normalized_tokens = []
        for token in tokens:
            # Remove WordPiece subword markers (##, ▁, etc.)
            normalized = re.sub(r'^##|^▁|^Ġ', '', token)
            # Remove special tokens markers
            normalized = re.sub(r'^<|>$', '', normalized)
            # Convert to lowercase for case-insensitive comparison
            normalized = normalized.lower().strip()
            # Skip empty tokens and very short tokens (likely artifacts)
            if normalized and len(normalized) > 0:
                normalized_tokens.append(normalized)
        
        return set(normalized_tokens)
    
    def calculate_recall(self, predicted: str, ground_truth) -> float:
        """Calculate token-based recall between predicted and ground truth texts."""
        pred_tokens = self.get_tokens(predicted)
        
        # Handle multiple ground truth answers
        if isinstance(ground_truth, list):
            recalls = []
            for truth in ground_truth:
                truth_tokens = self.get_tokens(truth)
                if truth_tokens:  # Only consider non-empty answers
                    intersection = truth_tokens.intersection(pred_tokens)
                    recall = len(intersection) / len(truth_tokens)
                    recalls.append(recall)
            return max(recalls) if recalls else 0.0
        else:
            # Single ground truth answer
            truth_tokens = self.get_tokens(ground_truth)
            if not truth_tokens:
                return 0.0
                
            # Calculate recall: |intersection| / |ground_truth|
            intersection = truth_tokens.intersection(pred_tokens)
            recall = len(intersection) / len(truth_tokens)
            
            return recall