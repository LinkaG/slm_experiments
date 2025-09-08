from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
from statistics import mean
from clearml import Task, Logger

from ..models.base import BaseModel
from ..retrievers.base import BaseRetriever
from ..data.base import BaseDataset
from .metrics import TokenRecallCalculator
from ..utils.memory_tracker import MemoryTracker
from ..utils.predictions_tracker import PredictionsTracker

@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    name: str
    model_config: Dict[str, Any]
    retriever_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    metrics_config: Dict[str, Any]
    output_dir: Path

class ExperimentRunner:
    """Main class for running experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metric_calculator = TokenRecallCalculator()
        self.memory_tracker = MemoryTracker(Path(config.output_dir))
        self.predictions_tracker = PredictionsTracker(Path(config.output_dir))
        
    def setup_experiment(self):
        """Initialize ClearML, create directories, etc."""
        self.task = Task.init(
            project_name="slm-experiments",
            task_name=self.config.name,
        )
        # Log configuration
        self.task.connect(self.config.__dict__)
        self.logger = Logger.current_logger()
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self, model: BaseModel, retriever: BaseRetriever, dataset: BaseDataset):
        """Run the experiment."""
        self.setup_experiment()
        
        # Initial memory state
        self.memory_tracker.log_memory("system", "experiment_start")
        
        # Log basic info
        self.logger.report_scalar(
            title="model",
            series="size",
            value=model.get_model_size(),
            iteration=0
        )
        self.logger.report_scalar(
            title="retriever",
            series="index_size",
            value=retriever.get_index_size(),
            iteration=0
        )
        # Log dataset stats
        for key, value in dataset.get_dataset_stats().items():
            self.logger.report_scalar(
                title="dataset",
                series=key,
                value=value,
                iteration=0
            )
        
        # Run evaluation
        metrics = self._evaluate(model, retriever, dataset)
        
        # Log results
        for metric_name, value in metrics.items():
            self.logger.report_scalar(
                title="metrics",
                series=metric_name,
                value=value,
                iteration=0
            )
        self._save_results(metrics)
        
        # Final memory state and save memory log
        self.memory_tracker.log_memory("system", "experiment_end")
        self.memory_tracker.save_log()
        
        # Save predictions and upload as artifact
        self.predictions_tracker.save_predictions()
        
    def _get_context(self, item: DatasetItem, retriever: Optional[BaseRetriever]) -> List[str]:
        """Get context based on experiment mode."""
        if not self.config.use_retriever:
            if self.config.context_type == "none":
                return []
            elif self.config.context_type == "oracle":
                return [item.context] if item.context else []
        else:
            if not retriever:
                raise ValueError("Retriever is required for retriever_context mode")
            contexts = retriever.retrieve(
                item.question, 
                top_k=self.config.top_k
            )
            return [c.context for c in contexts if c.score >= self.config.min_score]

    def _evaluate(self, model: BaseModel, retriever: Optional[BaseRetriever], 
                 dataset: BaseDataset) -> Dict[str, float]:
        """Evaluate model performance using token recall metric."""
        recalls = []
        processed = 0
        
        for item in dataset.get_eval_data():
            if not item.answer:  # Skip items without ground truth
                continue
            
            if self.config.max_samples and processed >= self.config.max_samples:
                break
                
            # Track memory for retrieval
            if retriever:
                self.memory_tracker.log_memory("retriever", "before_retrieve")
            
            # Get context based on experiment mode
            contexts = self._get_context(item, retriever)
            
            if retriever:
                self.memory_tracker.log_memory("retriever", "after_retrieve")
            
            # Track memory for model inference
            self.memory_tracker.log_memory("model", "before_generate")
            
            # Generate answer
            predicted_answer = model.generate(
                prompt=item.question,
                context=contexts
            )
            
            self.memory_tracker.log_memory("model", "after_generate")
            
            # Clear memory periodically
            if processed % 100 == 0:
                self.memory_tracker.clear_memory()
            
            # Calculate token recall
            recall = self.metric_calculator.calculate_recall(
                predicted=predicted_answer,
                ground_truth=item.answer
            )
            recalls.append(recall)
            
            # Track prediction
            self.predictions_tracker.add_prediction(
                question_id=item.metadata.get('id', str(processed)),
                question=item.question,
                predicted_answer=predicted_answer,
                ground_truth=item.answer,
                contexts=contexts,
                context_type=self.config.context_type,
                model_name=self.config.model.name,
                token_recall=recall,
                metadata={
                    'dataset': self.config.dataset.name,
                    **item.metadata
                }
            )
            
            # Log individual example
            self.logger.report_scalar(
                title="examples",
                series="token_recall",
                value=recall,
                iteration=len(recalls)
            )
        
        # Calculate average recall
        avg_recall = mean(recalls) if recalls else 0.0
        
        return {
            "token_recall": avg_recall,
            "num_examples": len(recalls)
        }
    
    def _save_results(self, metrics: Dict[str, float]):
        """Save experiment results to disk."""
        results_file = self.config.output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(metrics, f, indent=2)
