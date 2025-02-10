"""Offline RL trainer module."""
from typing import Dict, Optional, Union
import numpy as np
from pathlib import Path
import wandb
from src.models.cql_model import CQLModel  # Our custom CQL implementation

class OfflineRLTrainer:
    """Trainer class for offline RL models."""
    
    def __init__(
        self,
        algorithm: str = "cql",
        model_params: Optional[Dict] = None,
        experiment_name: str = "offline_rl_training"
    ):
        """Initialize the trainer.
        
        Args:
            algorithm: Name of the offline RL algorithm to use
            model_params: Parameters for the algorithm
            experiment_name: Name for MLflow experiment
        """
        self.algorithm = algorithm
        self.model_params = model_params or {}
        self.experiment_name = experiment_name
        
        # Initialize the algorithm
        if algorithm.lower() == "cql":
            self.model = CQLModel(**self.model_params)
        else:
            raise ValueError(f"Algorithm {algorithm} not supported")
        
        # Initialize W&B
        wandb.init(
            project=experiment_name,
            config={
                "algorithm": algorithm,
                **self.model_params
            }
        )
    
    def evaluate(self, eval_dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate the trained model.
        
        Args:
            eval_dataset: Dataset for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = self.model.evaluate(eval_dataset)
        wandb.log(metrics)
        return metrics
