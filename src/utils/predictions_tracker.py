from typing import Dict, List, Optional
import json
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
from clearml import Task

@dataclass
class PredictionItem:
    """Container for model prediction and related data."""
    question_id: str
    question: str
    predicted_answer: str
    ground_truth: Optional[str]
    contexts: List[str]  # использованные контексты
    context_type: str  # 'none', 'oracle', или 'retrieved'
    model_name: str
    token_recall: float
    metadata: Dict  # дополнительная информация

class PredictionsTracker:
    """Tracks and stores model predictions and related data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.predictions: List[PredictionItem] = []
        self.logger = logging.getLogger(__name__)
        
    def add_prediction(self, 
                      question_id: str,
                      question: str,
                      predicted_answer: str,
                      ground_truth: Optional[str],
                      contexts: List[str],
                      context_type: str,
                      model_name: str,
                      token_recall: float,
                      metadata: Optional[Dict] = None):
        """Add new prediction."""
        item = PredictionItem(
            question_id=question_id,
            question=question,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth,
            contexts=contexts,
            context_type=context_type,
            model_name=model_name,
            token_recall=token_recall,
            metadata=metadata or {}
        )
        self.predictions.append(item)
    
    def save_predictions(self):
        """Save predictions to file and upload to ClearML."""
        # Сохраняем локально
        predictions_file = self.output_dir / "predictions.json"
        predictions_data = {
            "predictions": [asdict(p) for p in self.predictions],
            "statistics": self._calculate_statistics()
        }
        
        with open(predictions_file, "w", encoding='utf-8') as f:
            json.dump(predictions_data, f, ensure_ascii=False, indent=2)
            
        self.logger.info(f"Predictions saved to {predictions_file}")
        
        # Загружаем в ClearML как артефакт
        task = Task.current_task()
        if task:
            task.upload_artifact(
                name="predictions",
                artifact_object=predictions_file,
                metadata={
                    "num_predictions": len(self.predictions),
                    "context_types": self._get_unique_context_types(),
                    "model_names": self._get_unique_model_names()
                }
            )
    
    def _calculate_statistics(self) -> Dict:
        """Calculate basic statistics about predictions."""
        if not self.predictions:
            return {}
            
        # Группируем метрики по типам контекста
        metrics_by_context = {}
        for pred in self.predictions:
            if pred.context_type not in metrics_by_context:
                metrics_by_context[pred.context_type] = {
                    "count": 0,
                    "total_recall": 0.0,
                    "has_ground_truth": 0
                }
            
            stats = metrics_by_context[pred.context_type]
            stats["count"] += 1
            stats["total_recall"] += pred.token_recall
            if pred.ground_truth:
                stats["has_ground_truth"] += 1
        
        # Вычисляем средние значения
        for context_type, stats in metrics_by_context.items():
            if stats["count"] > 0:
                stats["avg_recall"] = stats["total_recall"] / stats["count"]
                stats["ground_truth_ratio"] = stats["has_ground_truth"] / stats["count"]
                
        return {
            "total_predictions": len(self.predictions),
            "metrics_by_context": metrics_by_context,
            "unique_models": self._get_unique_model_names()
        }
    
    def _get_unique_context_types(self) -> List[str]:
        """Get list of unique context types."""
        return list(set(p.context_type for p in self.predictions))
    
    def _get_unique_model_names(self) -> List[str]:
        """Get list of unique model names."""
        return list(set(p.model_name for p in self.predictions))
