import os
import torch
import typing
import logging

import pytorch_lightning as pl

from src.model import MoLiNER
from pytorch_lightning.callbacks import Callback
from src.data.typing import RawBatch, EvaluationResult
from src.visualizations.spans import plot_evaluation_results

logger = logging.getLogger(__name__)

DEFAULT_DIR_NAME = "visualizations"

class VisualizationCallback(Callback):
    """
    Callback to generate and log visualizations of model predictions on a fixed validation batch.
    
    This callback runs at the beginning and/or end of each training epoch. It evaluates the
    model on a pre-selected batch of validation data and saves the prediction plots as HTML files.
    """
    def __init__(
        self,
        dirpath: str,
        batch_index: int = 0,
        num_samples: int = 2,
        score_threshold: float = 0.5,
        fps: int = 20,
        visualize_on_start: bool = True,
        visualize_on_end: bool = True,
    ):
        """
        Args:
            dirpath (str): The path were to save the visualizations at during training.
            batch_index (int): The index of the validation batch to use for visualization.
            num_samples (int): The number of samples from the batch to visualize.
            score_threshold (float): The confidence threshold for predictions.
            fps (int): Frames per second for the output visualization.
            visualize_on_start (bool): Whether to run visualization at the start of the epoch.
            visualize_on_end (bool): Whether to run visualization at the end of the epoch.
        """
        super().__init__()
        
        self.dirpath = dirpath
        self.batch_index = batch_index
        self.num_samples = num_samples
        self.score_threshold = score_threshold
        self.fps = fps
        self.visualize_on_start = visualize_on_start
        self.visualize_on_end = visualize_on_end
        self.visualization_batch: typing.Optional[RawBatch] = None
        
        os.makedirs(self.dirpath, exist_ok=True)

    # def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    def on_train_start(self, trainer: pl.Trainer, pl_module: 'MoLiNER'):
        """
        Fetches and stores the validation batch for later use.
        """
        if not (self.visualize_on_start or self.visualize_on_end):
            return

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
                self.visualization_batch = batch
                logger.info(f"Successfully stored validation batch {i} for visualization.")
                break

    # def _run_and_log_visualizations(self, trainer: pl.Trainer, pl_module: pl.LightningModule, when: str):
    def _run_and_log_visualizations(self, trainer: pl.Trainer, pl_module: 'MoLiNER', when: str):
        """
        Helper function to run evaluation and save plots.
        """
        if self.visualization_batch is None or trainer.logger is None:
            logger.warning("No visualization batch available or logger is not set. Skipping visualizations.")
            return

        logger.info(f"Generating visualizations for epoch {pl_module.current_epoch} ({when})...")
        
        original_mode = pl_module.training
        pl_module.eval()
        
        with torch.no_grad():
            raw_batch = self.visualization_batch
            num_to_visualize = min(self.num_samples, len(raw_batch.sid))

            for i in range(num_to_visualize):
                motion_length = int(raw_batch.motion_mask[i].sum())
                motion_tensor = raw_batch.transformed_motion[i, :motion_length, :]
                prompt_texts = [prompt[0] for prompt in raw_batch.prompts[i]]

                if not prompt_texts:
                    logger.warning(f"Sample {i} in visualization batch has no prompts. Skipping.")
                    continue
                
                evaluation_result = pl_module.evaluate(
                    motion=motion_tensor,
                    prompts=prompt_texts,
                    score_threshold=self.score_threshold,
                )
                
                prediction_figure = plot_evaluation_results(
                    evaluation_result,
                    title=f"Epoch {pl_module.current_epoch} ({when}) - Sample {i}",
                    fps=self.fps
                )
                if prediction_figure:
                    filename = f"epoch_{pl_module.current_epoch}_{when}_sample_{i}_prediction.html"
                    output_path = os.path.join(self.dirpath, filename)
                    prediction_figure.write_html(output_path)
                    logger.info(f"Saved visualization to {os.path.abspath(output_path)}")
                else:
                    logger.warning(f"No predictions to visualize for sample {i} in epoch {pl_module.current_epoch} ({when}).")
                
                groundtruth_spans = [(p[0], s[0], s[1], 1.0) for p in raw_batch.prompts[i] for s in p[1]]
                groundtruth_result = EvaluationResult(motion_length=motion_length, predictions=groundtruth_spans)
                groundtruth_figure = plot_evaluation_results(
                    groundtruth_result,
                    title=f"Epoch {pl_module.current_epoch} ({when}) - Sample {i} (Ground Truth)",
                    fps=self.fps
                )
                if groundtruth_figure:
                    gt_filename = f"epoch_{pl_module.current_epoch}_{when}_sample_{i}_groundtruth.html"
                    gt_output_path = os.path.join(self.dirpath, gt_filename)
                    groundtruth_figure.write_html(gt_output_path)
                    logger.info(f"Saved ground truth visualization to {os.path.abspath(gt_output_path)}")
                else:
                    logger.warning(f"No ground truth spans to visualize for sample {i} in epoch {pl_module.current_epoch} ({when}).")
                    
        pl_module.train(original_mode)
        
    # def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: 'MoLiNER'):
        if trainer.sanity_checking:
            return
        if self.visualize_on_start:
            self._run_and_log_visualizations(trainer, pl_module, when="start")

    # def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: 'MoLiNER'):
        if trainer.sanity_checking:
            return
        if self.visualize_on_end:
            self._run_and_log_visualizations(trainer, pl_module, when="end")