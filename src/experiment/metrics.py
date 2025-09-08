from typing import List, Set
from transformers import AutoTokenizer

class TokenRecallCalculator:
    """Calculator for token-based recall metric."""
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        """Initialize with specified tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def get_tokens(self, text: str) -> Set[str]:
        """Convert text to a set of tokens."""
        # Tokenize and convert to text (to handle special tokens)
        tokens = self.tokenizer.tokenize(text)
        # Remove special tokens and convert to set
        return set(token.replace("##", "") for token in tokens)
    
    def calculate_recall(self, predicted: str, ground_truth: str) -> float:
        """Calculate token-based recall between predicted and ground truth texts."""
        pred_tokens = self.get_tokens(predicted)
        truth_tokens = self.get_tokens(ground_truth)
        
        if not truth_tokens:
            return 0.0
            
        # Calculate recall: |intersection| / |ground_truth|
        intersection = pred_tokens.intersection(truth_tokens)
        recall = len(intersection) / len(truth_tokens)
        
        return recall
