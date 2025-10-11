from typing import List, Set
from transformers import AutoTokenizer

class TokenRecallCalculator:
    """Calculator for token-based recall metric."""
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        """Initialize with specified tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def get_tokens(self, text) -> Set[str]:
        """Convert text to a set of tokens."""
        # Handle list of answers
        if isinstance(text, list):
            text = " ".join(text)
        
        # Ensure text is string
        if not isinstance(text, str):
            text = str(text)
            
        # Tokenize and convert to text (to handle special tokens)
        tokens = self.tokenizer.tokenize(text)
        # Remove special tokens and convert to set
        return set(token.replace("##", "") for token in tokens)
    
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