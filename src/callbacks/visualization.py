import os
import torch
import logging
import typing

import numpy as np
import pytorch_lightning as pl

from src.model import MoLiNER
from pytorch_lightning.callbacks import Callback
from src.types import RawBatch
from src.visualizations.generator import VisualizationGenerator

logger = logging.getLogger(__name__)

class VisualizationCallback(Callback):
    """
    Simplified callback to generate and log visualizations of model predictions on a fixed validation batch.
    
    This callback runs at the end of each training epoch using the new batch-based approach.
    """
    def __init__(
        self,
        dirpath: str,
        batch_index: int,
        score_threshold: typing.Union[float, typing.List[float]],
        debug: bool = False,
        visualization_frequency: int = 1,
        skip_html_generation: bool = False
    ):
        """
        Args:
            dirpath (str): The path where to save the visualizations during training.
            batch_index (int): The index of the validation batch to use for visualization.
            score_threshold (float or List[float]): The confidence threshold(s) for predictions.
            debug (bool): Whether to output debug information.
            visualization_frequency (int): Run visualization every N epochs (default: 1).
            skip_html_generation (bool): Skip HTML generation for faster execution (default: False).
        """
        super().__init__()
        
        self.dirpath = dirpath
        self.batch_index = batch_index
        
        if isinstance(score_threshold, (int, float)):
            score_thresholds = [float(score_threshold)]
        else:
            score_thresholds = [float(t) for t in score_threshold]
        
        self.visualization_generator = VisualizationGenerator(
            score_thresholds=score_thresholds,
            debug=debug
        )
        
        self.visualization_batch: typing.Optional["RawBatch"] = None
        self.debug = debug
        self.visualization_frequency = max(1, visualization_frequency)
        self.skip_html_generation = skip_html_generation
        
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Fetches and stores the validation batch for later use.
        """
        if self.debug:
            logger.info(f"Setting up visualization: fetching validation batch index {self.batch_index}.")
        
        validation_dataloaders = trainer.val_dataloaders
        if not validation_dataloaders:
            logger.warning("No validation dataloader found. Cannot perform epoch-wise visualizations.")
            return

        validation_dataloader = validation_dataloaders[0] if isinstance(validation_dataloaders, list) else validation_dataloaders
        
        if len(validation_dataloader) <= self.batch_index:
            logger.warning(
                f"visualization_batch_index ({self.batch_index}) is out of bounds "
                f"for the validation dataloader (size: {len(validation_dataloader)}). Disabling visualizations."
            )
            return

        for i, batch in enumerate(validation_dataloader):
            if i == self.batch_index:
                self.visualization_batch = self._move_batch_to_cpu(batch)
                if self.debug:
                    logger.info(f"Successfully stored validation batch {i} for visualization.")
                    logger.info(f"Batch contains {len(batch.sid)} samples.")
                break
                
    def _move_batch_to_cpu(self, batch: "RawBatch") -> "RawBatch":
        """Move batch tensors to CPU to avoid device issues."""
        return RawBatch(
            sid=batch.sid,
            dataset_name=batch.dataset_name,
            amass_relative_path=batch.amass_relative_path,
            raw_motion=batch.raw_motion.detach().cpu() if batch.raw_motion is not None else torch.empty(0),
            transformed_motion=batch.transformed_motion.detach().cpu(),
            motion_mask=batch.motion_mask.detach().cpu(),
            prompts=batch.prompts
        )

    def _run_and_log_visualizations(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Generate and save visualizations using the simplified batch-based approach."""
        from src.model.moliner import MoLiNER
        
        if self.visualization_frequency > 1 and pl_module.current_epoch % self.visualization_frequency != 0:
            return
            
        if not isinstance(pl_module, MoLiNER):
            logger.error(f"Expected MoLiNER model but got {type(pl_module)}")
            return
            
        model = pl_module
        
        if self.visualization_batch is None or trainer.logger is None:
            logger.warning("No visualization batch available or logger is not set. Skipping visualizations.")
            return

        if self.debug:
            logger.info(f"Generating visualizations for epoch {model.current_epoch}")
        
        # Generate visualization data for the entire batch at once
        visualization_results = self.visualization_generator.generate_visualization_data(
            model=model,
            raw_batch=self.visualization_batch,
            epoch=model.current_epoch
        )
        
        if not self.skip_html_generation:
            self._save_visualizations(visualization_results, trainer)
        else:
            if self.debug:
                logger.info(f"Skipping HTML generation for faster execution")
                
    def _save_visualizations(self, visualization_results, trainer: pl.Trainer):
        """Save visualizations to disk and log to WandB if available."""
        wandb_logger = self._get_wandb_logger(trainer)
        
        epoch = visualization_results.epoch
        epoch_dir = os.path.join(self.dirpath, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        for i, sample in visualization_results.samples.items():
            try:
                filename = f"sample_{i}.html"
                output_path = os.path.join(epoch_dir, filename)
                
                title = f"Epoch {epoch} - Sample {i} - Combined GT & All Threshold Predictions"
                
                combined_figure = self.visualization_generator.create_combined_visualization(
                    sample=sample,
                    title=title,
                    output_path=output_path
                )
                
                if combined_figure and wandb_logger is not None:
                    try:
                        # type: ignore
                        wandb_logger.log_html_visualization(
                            html_path=output_path,
                            key=f"visualizations/epoch_{epoch:03d}/sample_{i}"
                        )
                    except Exception as e:
                        if self.debug:
                            logger.warning(f"Failed to log combined visualization to WandB: {e}")
                    
            except Exception as e:
                logger.error(f"Error saving visualization for sample {i}: {e}")
                continue
    
    def _get_wandb_logger(self, trainer: pl.Trainer):
        """Extract WandB logger if available."""
        wandb_logger = None
        if trainer.logger is not None:
            if hasattr(trainer, 'loggers') and trainer.loggers:
                for logger_instance in trainer.loggers:
                    if hasattr(logger_instance, '__class__') and 'WandBLogger' in str(logger_instance.__class__):
                        wandb_logger = logger_instance
                        break
            elif hasattr(trainer.logger, '__class__') and 'WandBLogger' in str(trainer.logger.__class__):
                wandb_logger = trainer.logger
        return wandb_logger
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Run visualizations at the end of each training epoch."""
        if trainer.sanity_checking:
            return
        self._run_and_log_visualizations(trainer, pl_module)