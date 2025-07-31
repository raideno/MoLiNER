import re
import os
import torch
import wandb
import typing
import logging

from argparse import Namespace

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)

class WandBLogger(Logger):
    """
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
        name: typing.Optional[str] = None,
        entity: typing.Optional[str] = None,
        tags: typing.Optional[list] = None,
        notes: typing.Optional[str] = None,
        save_dir: typing.Optional[str] = None,
        offline: bool = False,
        config: typing.Optional[typing.Dict[str, typing.Any]] = None,
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
        
        logger.info(f"[wandb](tags): {self._tags}")

    @property
    def name(self) -> typing.Optional[str]:
        return self._name

    @property
    def version(self) -> typing.Optional[str]:
        if self._experiment is not None:
            return self._experiment.id
        return None

    @property
    def experiment(self) -> typing.Optional[typing.Any]:
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

    # NOTE: ensures the method only runs on the main process, in case of a distributed setup
    @rank_zero_only
    def log_hyperparams(self, params: typing.Union[typing.Dict[str, typing.Any], Namespace, DictConfig], *args: typing.Any, **kwargs: typing.Any) -> None:
        processed_params: typing.Dict[str, typing.Any] = {}
        
        if isinstance(params, Namespace):
            processed_params = vars(params)
        elif isinstance(params, DictConfig):
            container = OmegaConf.to_container(params, resolve=True)
            if isinstance(container, dict):
                # processed_params = container
                processed_params = {str(k): v for k, v in container.items()}
        elif isinstance(params, dict):
            processed_params = params
        
        self._config.update(processed_params)
        
        exp = self.experiment
        if self._initialized and exp is not None:
            exp.config.update(processed_params)

    @rank_zero_only
    def log_metrics(self, metrics: typing.Dict[str, float], step: typing.Optional[int] = None) -> None:
        exp = self.experiment
        if exp is None:
            return
        
        processed_metrics: typing.Dict[str, float] = {}
        # pyrefly: ignore
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
        
        exp.log(processed_metrics)
        # NOTE: commented to let W&B manage steps & thus avoid step synchronization issues
        # exp.log(processed_metrics, step=step)

    @rank_zero_only
    def log_model_config(self, config: typing.Dict[str, typing.Any]) -> None:
        self.log_hyperparams({"model_config": config})

    @rank_zero_only
    def log_artifacts(self, artifact_path: str, artifact_type: str = "model") -> None:
        exp = self.experiment
        if exp is None:
            return
            
        try:
            sanitized_name = self._sanitize_artifact_name(f"{self._name}_{artifact_type}")
            
            artifact = wandb.Artifact(name=sanitized_name, type=artifact_type)
            artifact.add_dir(artifact_path)
            exp.log_artifact(artifact)
        except Exception as e:
            logger.warning(f"Failed to log artifact {artifact_path}: {e}")

    def _sanitize_artifact_name(self, name: str) -> str:
        sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
        sanitized = re.sub(r'_+', '_', sanitized)
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
        exp = self.experiment
        if exp is None:
            return
            
        try:
            exp.log({key: wandb.Html(html_path)})
        except Exception as e:
            logger.warning(f"Failed to log HTML visualization {key}: {e}")

    @rank_zero_only
    def save_config_as_json(self, config_dict: typing.Dict[str, typing.Any], save_path: str) -> None:
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
            pass