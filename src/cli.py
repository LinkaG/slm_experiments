import typer
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging

from .experiment.runner import ExperimentRunner, ExperimentConfig
from .models import get_model
from .retrievers import get_retriever
from .data import get_dataset

app = typer.Typer()

@app.command()
@hydra.main(config_path="../configs", config_name="config")
def run_experiment(cfg: DictConfig):
    """Run an experiment with specified configuration."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create experiment config
    config = ExperimentConfig(
        name=cfg.experiment.name,
        model_config=dict(cfg.model),
        retriever_config=dict(cfg.retriever),
        dataset_config=dict(cfg.dataset),
        metrics_config=dict(cfg.metrics),
        output_dir=Path(cfg.experiment.output_dir)
    )
    
    # Initialize components
    model = get_model(cfg.model)
    retriever = get_retriever(cfg.retriever)
    dataset = get_dataset(cfg.dataset)
    
    # Run experiment
    runner = ExperimentRunner(config)
    runner.run(model, retriever, dataset)
    
    logger.info(f"Experiment {config.name} completed successfully!")

if __name__ == "__main__":
    app()
