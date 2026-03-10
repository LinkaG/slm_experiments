from typing import List, Set, Optional, Any, Union
import re

from transformers import AutoTokenizer


def get_ground_truth_for_recall(metadata: Optional[dict], answer: Any) -> Union[str, List[str], None]:
    """
    Возвращает ground truth для расчёта recall.
    MIRAGE: использует all_answers (лучший recall по токенам), иначе — одиночный ответ.
    """
    return (metadata or {}).get('all_answers') or answer


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
    
    def _get_words(self, text) -> List[str]:
        """Разбить текст на слова."""
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        if not isinstance(text, str):
            text = str(text)
        if not text.strip():
            return []
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w]
    
    def calculate_token_recall(self, predicted: str, ground_truth) -> float:
        """
        Recall по токенам (оригинальный): intersection / |ground_truth_tokens|.
        Для all_answers — берём max.
        """
        pred_tokens = self.get_tokens(predicted)
        if isinstance(ground_truth, list):
            recalls = []
            for truth in ground_truth:
                truth_tokens = self.get_tokens(truth)
                if truth_tokens:
                    intersection = truth_tokens.intersection(pred_tokens)
                    recalls.append(len(intersection) / len(truth_tokens))
            return max(recalls) if recalls else 0.0
        else:
            truth_tokens = self.get_tokens(ground_truth)
            if not truth_tokens:
                return 0.0
            intersection = truth_tokens.intersection(pred_tokens)
            return len(intersection) / len(truth_tokens)
    
    def _recall_substring(self, predicted: str, truth: str) -> float:
        """
        Recall по словам: каждое слово ground truth ищем как подстроку в предсказании.
        recall = доля найденных слов.
        """
        pred_norm = predicted.lower()
        truth_words = self._get_words(truth)
        if not truth_words:
            return 0.0
        matched = sum(1 for w in truth_words if w in pred_norm)
        return matched / len(truth_words)
    
    def calculate_recall(self, predicted: str, ground_truth) -> float:
        """
        Recall по словам как подстрокам: разбить ground truth на слова,
        для каждого проверить, есть ли оно как подстрока в предсказании.
        recall = доля найденных слов. Для all_answers — берём max.
        """
        # Handle multiple ground truth answers (all_answers)
        if isinstance(ground_truth, list):
            recalls = []
            for truth in ground_truth:
                if truth:
                    r = self._recall_substring(predicted, truth)
                    recalls.append(r)
            return max(recalls) if recalls else 0.0
        else:
            return self._recall_substring(predicted, ground_truth)