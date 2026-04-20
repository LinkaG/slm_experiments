from typing import Dict, List, Optional, Any
import json
from pathlib import Path
from dataclasses import dataclass
import logging
from clearml import Task

@dataclass
class PredictionItem:
    """Container for model prediction and related data."""
    question_id: str
    question: str
    predicted_answer: str
    ground_truth: Optional[Any]
    contexts: List[str]
    context_type: str
    model_name: str
    token_recall: float
    metadata: Dict
    prompt: Optional[str] = None

class PredictionsTracker:
    """Tracks and stores model predictions; сохраняет только компактный JSON для артефактов."""

    def __init__(self, output_dir: Path, experiment_name: str):
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self._stem = experiment_name.replace("/", "_").replace("\\", "_")
        self.predictions: List[PredictionItem] = []
        self.logger = logging.getLogger(__name__)

    @property
    def predictions_json_path(self) -> Path:
        return self.output_dir / f"{self._stem}_predictions.json"

    def add_prediction(self,
                      question_id: str,
                      question: str,
                      predicted_answer: str,
                      ground_truth: Optional[Any],
                      contexts: List[str],
                      context_type: str,
                      model_name: str,
                      token_recall: float,
                      metadata: Optional[Dict] = None,
                      prompt: Optional[str] = None):
        item = PredictionItem(
            question_id=question_id,
            question=question,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth,
            contexts=contexts,
            context_type=context_type,
            model_name=model_name,
            token_recall=token_recall,
            metadata=metadata or {},
            prompt=prompt
        )
        self.predictions.append(item)

    def save_predictions(self, upload_to_minio_callback=None):
        """
        Файл {experiment}_predictions.json: question_id, question, ground_truth, predicted_answer.
        """
        path = self.predictions_json_path
        rows = [
            {
                "question_id": p.question_id,
                "question": p.question,
                "ground_truth": p.ground_truth,
                "predicted_answer": p.predicted_answer,
            }
            for p in self.predictions
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Predictions saved to {path}")

        if upload_to_minio_callback:
            s3_key = f"experiment_results/{self._stem}_predictions.json"
            s3_path = upload_to_minio_callback(path, s3_key)
            if s3_path:
                self.logger.info(f"✅ Predictions uploaded to MinIO: {s3_path}")
            return s3_path
        task = Task.current_task()
        if task:
            task.upload_artifact(
                name="predictions",
                artifact_object=path,
                metadata={"num_predictions": len(self.predictions)},
            )
        return None

    def _calculate_statistics(self) -> Dict:
        if not self.predictions:
            return {}
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
        for context_type, stats in metrics_by_context.items():
            if stats["count"] > 0:
                stats["avg_recall"] = stats["total_recall"] / stats["count"]
                stats["ground_truth_ratio"] = stats["has_ground_truth"] / stats["count"]
        return {
            "total_predictions": len(self.predictions),
            "metrics_by_context": metrics_by_context,
            "unique_models": self._get_unique_model_names()
        }

    def get_statistics(self) -> Dict:
        return self._calculate_statistics()

    def _get_unique_context_types(self) -> List[str]:
        return list(set(p.context_type for p in self.predictions))

    def _get_unique_model_names(self) -> List[str]:
        return list(set(p.model_name for p in self.predictions))
