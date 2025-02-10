"""Model deployment and serving module."""
from typing import Dict, Optional, Any
import os
from datetime import datetime
import json
import mlflow
from pathlib import Path

class ModelDeployer:
    """Handles model deployment and versioning."""
    
    def __init__(
        self,
        deployment_dir: str = "deployments",
        config_dir: str = "config"
    ):
        """Initialize the deployer.
        
        Args:
            deployment_dir: Directory for model deployments
            config_dir: Directory for deployment configs
        """
        self.deployment_dir = Path(deployment_dir)
        self.config_dir = Path(config_dir)
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
    def deploy_model(
        self,
        model_path: str,
        model_version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Deploy a new model version.
        
        Args:
            model_path: Path to the trained model
            model_version: Version identifier for the model
            metadata: Additional metadata about the deployment
            
        Returns:
            Path to the deployed model
        """
        # Create deployment timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create deployment directory
        deploy_path = self.deployment_dir / f"model_v{model_version}_{timestamp}"
        deploy_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model to deployment directory
        new_model_path = deploy_path / "model.pt"
        os.system(f"cp {model_path} {new_model_path}")
        
        # Save deployment config
        config = {
            "model_version": model_version,
            "deployment_timestamp": timestamp,
            "original_model_path": model_path,
            "metadata": metadata or {}
        }
        
        config_path = deploy_path / "deployment_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        return str(deploy_path)
    
    def load_deployed_model(self, version: str) -> str:
        """Load a specific deployed model version.
        
        Args:
            version: Version of the model to load
            
        Returns:
            Path to the model file
        """
        # Find the latest deployment for the specified version
        deployments = list(self.deployment_dir.glob(f"model_v{version}_*"))
        if not deployments:
            raise ValueError(f"No deployment found for version {version}")
        
        latest_deployment = max(deployments, key=lambda x: x.stat().st_mtime)
        model_path = latest_deployment / "model.pt"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found in deployment {latest_deployment}")
        
        return str(model_path)
    
    def list_deployments(self) -> Dict[str, Dict]:
        """List all deployed models.
        
        Returns:
            Dictionary of deployment information
        """
        deployments = {}
        for deploy_dir in self.deployment_dir.glob("model_v*"):
            config_path = deploy_dir / "deployment_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                deployments[deploy_dir.name] = config
        
        return deployments
    
    def rollback(self, version: str) -> str:
        """Rollback to a specific model version.
        
        Args:
            version: Version to rollback to
            
        Returns:
            Path to the rolled back model
        """
        model_path = self.load_deployed_model(version)
        
        # Create a new deployment with rollback metadata
        metadata = {
            "rollback": True,
            "rollback_from": version,
            "rollback_timestamp": datetime.now().isoformat()
        }
        
        return self.deploy_model(
            model_path=model_path,
            model_version=f"{version}_rollback",
            metadata=metadata
        )
