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
        
        Note: If neither tokenizer nor tokenizer_name is provided, the calculator
        will work without a tokenizer. The get_tokens() method uses regex and doesn't
        require a tokenizer, so this is safe.
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            # Don't load tokenizer if not needed - get_tokens() uses regex only
            # This avoids unnecessary downloads and dependency issues
            self.tokenizer = None
    
    def get_tokens(self, text) -> Set[str]:
        """
        Convert text to a set of normalized word tokens.
        Tokens are individual words (split by whitespace and punctuation).
        
        Args:
            text: Input text (str, list, or other)
            
        Returns:
            Set of normalized word strings
        """
        # Handle list of answers
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        
        # Ensure text is string
        if not isinstance(text, str):
            text = str(text)
        
        if not text.strip():
            return set()
        
        # Extract words: sequences of letters, digits, and common word characters
        # This splits on whitespace and punctuation, keeping only words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out empty strings and return unique words
        return set(word for word in words if word)
    
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