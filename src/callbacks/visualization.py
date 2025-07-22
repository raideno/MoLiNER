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
    Callback to generate and log visualizations of model predictions on a fixed validation batch.
    
    This callback runs at the beginning and/or end of each training epoch. It evaluates the
    model on a pre-selected batch of validation data and saves the prediction plots as HTML files.
    """
    def __init__(
        self,
        dirpath: str,
        batch_index: int,
        num_samples: int,
        score_threshold: typing.Union[float, typing.List[float]],
        debug: bool = False,
        visualization_frequency: int = 1,
        skip_html_generation: bool = False
    ):
        """
        Args:
            dirpath (str): The path were to save the visualizations at during training.
            batch_index (int): The index of the validation batch to use for visualization.
            num_samples (int): The number of samples from the batch to visualize.
            score_threshold (float or List[float]): The confidence threshold(s) for predictions.
                Can be a single float or a list of floats for multiple thresholds.
            visualization_frequency (int): Run visualization every N epochs (default: 1).
            skip_html_generation (bool): Skip HTML generation for faster execution (default: False).
        """
        super().__init__()
        
        self.dirpath = dirpath
        self.batch_index = batch_index
        self.num_samples = num_samples
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
                # NOTE: we store the batch and move tensors to CPU to avoid device issues
                self.visualization_batch = self._move_batch_to_cpu(batch)
                if self.debug:
                    logger.info(f"Successfully stored validation batch {i} for visualization.")
                    logger.info(f"Batch contains {len(batch.sid)} samples with {len(batch.prompts[0])} prompts each.")
                break
                
    def _move_batch_to_cpu(self, batch: "RawBatch") -> "RawBatch":
        return RawBatch(
            sid=batch.sid,
            dataset_name=batch.dataset_name,
            amass_relative_path=batch.amass_relative_path,
            raw_motion=batch.raw_motion.cpu() if batch.raw_motion is not None else torch.empty(0),
            transformed_motion=batch.transformed_motion.cpu(),
            motion_mask=batch.motion_mask.cpu(),
            prompts=batch.prompts
        )

    def _run_and_log_visualizations(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
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
        

        num_to_visualize = min(self.num_samples, len(self.visualization_batch.sid))
        subset_batch = self._create_subset_batch(self.visualization_batch, num_to_visualize)
        
        visualization_results = self.visualization_generator.generate_visualization_data(
            model=model,
            raw_batch=subset_batch,
            epoch=model.current_epoch
        )
        
        if not self.skip_html_generation:
            self._save_visualizations_async(visualization_results, trainer)
        else:
            if self.debug:
                logger.info(f"Skipping HTML generation for faster execution")
                
    def _save_visualizations_async(self, visualization_results, trainer: pl.Trainer):
        wandb_logger = None
        if trainer.logger is not None:
            if hasattr(trainer, 'loggers') and trainer.loggers:
                for logger_instance in trainer.loggers:
                    if hasattr(logger_instance, '__class__') and 'WandBLogger' in str(logger_instance.__class__):
                        wandb_logger = logger_instance
                        break
            elif hasattr(trainer.logger, '__class__') and 'WandBLogger' in str(trainer.logger.__class__):
                wandb_logger = trainer.logger
        
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
        
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.sanity_checking:
            return
        self._run_and_log_visualizations(trainer, pl_module)
    
    def _create_subset_batch(self, batch: "RawBatch", num_samples: int) -> "RawBatch":
        return RawBatch(
            sid=batch.sid[:num_samples],
            dataset_name=batch.dataset_name[:num_samples],
            amass_relative_path=batch.amass_relative_path[:num_samples],
            raw_motion=batch.raw_motion[:num_samples] if batch.raw_motion is not None else None,
            transformed_motion=batch.transformed_motion[:num_samples],
            motion_mask=batch.motion_mask[:num_samples],
            prompts=batch.prompts[:num_samples]
        )