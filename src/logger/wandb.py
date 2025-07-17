import os
import torch
import wandb
import logging

from argparse import Namespace
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)

class WandBLogger(Logger):
    """WandB Logger for PyTorch Lightning.
    
    This logger saves model configuration, hyperparameters, and training metrics
    to Weights & Biases (WandB) for experiment tracking and visualization.
    
    Args:
        project: WandB project name
        name: Experiment name (run name in WandB)
        entity: WandB entity (team/user)
        tags: List of tags for the run
        notes: Notes for the run
        save_dir: Directory to save WandB logs locally
        offline: Whether to run in offline mode
        config: Configuration dictionary to log
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self._project = project
        self._name = name
        self._entity = entity
        self._tags = tags or []
        self._notes = notes
        self._save_dir = save_dir
        self._offline = offline
        self._config = config or {}
        
        self._experiment = None
        self._initialized = False

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def version(self) -> Optional[str]:
        if self._experiment is not None:
            return self._experiment.id
        return None

    @property
    def experiment(self) -> Optional[Any]:
        if self._experiment is not None:
            return self._experiment
        
        if self._offline:
            os.environ["WANDB_MODE"] = "offline"
        
        self._experiment = wandb.init(
            project=self._project,
            name=self._name,
            entity=self._entity,
            tags=self._tags,
            notes=self._notes,
            dir=self._save_dir,
            config=self._config,
            reinit=True,
        )
        
        self._initialized = True
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace, DictConfig], *args: Any, **kwargs: Any) -> None:
        """
        Log hyperparameters to WandB.
        """
        processed_params: Dict[str, Any] = {}
        
        if isinstance(params, Namespace):
            processed_params = vars(params)
        elif isinstance(params, DictConfig):
            container = OmegaConf.to_container(params, resolve=True)
            if isinstance(container, dict):
                processed_params = container
        elif isinstance(params, dict):
            processed_params = params
        
        self._config.update(processed_params)
        
        exp = self.experiment
        if self._initialized and exp is not None:
            exp.config.update(processed_params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to WandB.
        """
        exp = self.experiment
        if exp is None:
            return
        
        processed_metrics: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = float(value.item())
            elif isinstance(value, (int, float)):
                processed_metrics[key] = float(value)
            else:
                try:
                    processed_metrics[key] = float(value)
                except (TypeError, ValueError):
                    logger.warning(f"Could not convert metric {key} to float: {value}")
                    continue
        
        # Log without step parameter to avoid step synchronization issues
        # Let WandB handle step management automatically
        exp.log(processed_metrics)


    @rank_zero_only
    def log_model_config(self, config: Dict[str, Any]) -> None:
        self.log_hyperparams({"model_config": config})

    @rank_zero_only
    def log_artifacts(self, artifact_path: str, artifact_type: str = "model") -> None:
        """Log artifacts to WandB."""
        exp = self.experiment
        if exp is None:
            return
            
        try:
            # Sanitize artifact name to only contain valid characters
            sanitized_name = self._sanitize_artifact_name(f"{self._name}_{artifact_type}")
            
            artifact = wandb.Artifact(name=sanitized_name, type=artifact_type)
            artifact.add_dir(artifact_path)
            exp.log_artifact(artifact)
        except Exception as e:
            logger.warning(f"Failed to log artifact {artifact_path}: {e}")

    def _sanitize_artifact_name(self, name: str) -> str:
        """Sanitize artifact name to only contain valid characters."""
        import re
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized

    @rank_zero_only
    def watch_model(self, model: torch.nn.Module, log_freq: int = 1000) -> None:
        exp = self.experiment
        if exp is None:
            return
            
        try:
            wandb.watch(model, log_freq=log_freq)
        except Exception as exception:
            logger.warning(f"Failed to watch model: {exception}")

    @rank_zero_only
    def finish(self) -> None:
        if self._experiment is not None:
            try:
                self._experiment.finish()
            except Exception as e:
                logger.warning(f"Error finishing WandB run: {e}")
            finally:
                self._experiment = None
                self._initialized = False

    @rank_zero_only
    def log_html_visualization(self, html_path: str, key: str) -> None:
        """Log an HTML visualization to WandB."""
        exp = self.experiment
        if exp is None:
            return
            
        try:
            exp.log({key: wandb.Html(html_path)})
        except Exception as e:
            logger.warning(f"Failed to log HTML visualization {key}: {e}")

    @rank_zero_only
    def save_config_as_json(self, config_dict: Dict[str, Any], save_path: str) -> None:
        """Save configuration as JSON file."""
        try:
            import json
            from omegaconf import OmegaConf
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Convert OmegaConf to regular dict if needed
            if hasattr(config_dict, '__dict__') or hasattr(config_dict, '_content'):
                config_dict = OmegaConf.to_container(config_dict, resolve=True)
            
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save config as JSON: {e}")

    def __del__(self):
        try:
            if self._initialized:
                self.finish()
        except Exception:
            # Ignore errors during cleanup
            pass