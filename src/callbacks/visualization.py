import os
import torch
import logging
import typing

import numpy as np
import pytorch_lightning as pl

from src.model import MoLiNER
from pytorch_lightning.callbacks import Callback
from src.types import RawBatch, EvaluationResult
from src.visualizations.spans import plot_evaluation_results

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
        batch_index: int = 0,
        num_samples: int = 2,
        score_threshold: float = 0.5,
        visualize_on_start: bool = True,
        visualize_on_end: bool = True,
        debug: bool = False
    ):
        """
        Args:
            dirpath (str): The path were to save the visualizations at during training.
            batch_index (int): The index of the validation batch to use for visualization.
            num_samples (int): The number of samples from the batch to visualize.
            score_threshold (float): The confidence threshold for predictions.
            visualize_on_start (bool): Whether to run visualization at the start of the epoch.
            visualize_on_end (bool): Whether to run visualization at the end of the epoch.
        """
        super().__init__()
        
        self.dirpath = dirpath
        self.batch_index = batch_index
        self.num_samples = num_samples
        self.score_threshold = score_threshold
        self.visualize_on_start = visualize_on_start
        self.visualize_on_end = visualize_on_end
        self.visualization_batch: typing.Optional[RawBatch] = None
        self.debug = debug
        
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Fetches and stores the validation batch for later use.
        """
        if not (self.visualize_on_start or self.visualize_on_end):
            return

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
                
    def _move_batch_to_cpu(self, batch: RawBatch) -> RawBatch:
        """
        Move batch tensors to CPU for storage.
        """
        return RawBatch(
            sid=batch.sid,
            dataset_name=batch.dataset_name,
            amass_relative_path=batch.amass_relative_path,
            raw_motion=batch.raw_motion.cpu() if batch.raw_motion is not None else torch.empty(0),
            transformed_motion=batch.transformed_motion.cpu(),
            motion_mask=batch.motion_mask.cpu(),
            prompts=batch.prompts
        )

    def _run_and_log_visualizations(self, trainer: pl.Trainer, pl_module: pl.LightningModule, when: str):
        """
        Helper function to run evaluation and save plots.
        """
        from src.model.moliner import MoLiNER
        
        if not isinstance(pl_module, MoLiNER):
            logger.error(f"Expected MoLiNER model but got {type(pl_module)}")
            return
            
        model = pl_module
        
        if self.visualization_batch is None or trainer.logger is None:
            logger.warning("No visualization batch available or logger is not set. Skipping visualizations.")
            return

        if self.debug:
            logger.info(f"Generating visualizations for epoch {model.current_epoch} ({when})...")
        
        original_mode = model.training
        model.eval()
        
        with torch.no_grad():
            raw_batch = self.visualization_batch
            num_to_visualize = min(self.num_samples, len(raw_batch.sid))

            for i in range(num_to_visualize):
                motion_length = int(raw_batch.motion_mask[i].sum())
                motion_tensor = raw_batch.transformed_motion[i, :motion_length, :].to(model.device)
                prompt_texts = [prompt[0] for prompt in raw_batch.prompts[i]]

                if not prompt_texts:
                    logger.warning(f"Sample {i} in visualization batch has no prompts. Skipping.")
                    continue
                
                if self.debug:
                    logger.info(f"Epoch {model.current_epoch} ({when}) - Sample {i}: Motion tensor shape: {motion_tensor.shape}, Device: {motion_tensor.device}")
                    logger.info(f"Epoch {model.current_epoch} ({when}) - Sample {i}: Model device: {model.device}")
                    logger.info(f"Epoch {model.current_epoch} ({when}) - Sample {i}: Prompts: {prompt_texts}")
                
                evaluation_result = model.evaluate(
                    motion=motion_tensor,
                    prompts=prompt_texts,
                    score_threshold=self.score_threshold,
                )
                
                result_filename = f"epoch_{model.current_epoch}_{when}_sample_{i}_evaluation_result.pt"
                result_path = os.path.join(self.dirpath, result_filename)
                
                result_data = {
                    "motion_length": int(evaluation_result.motion_length),
                    "predictions": evaluation_result.predictions if evaluation_result.predictions else [],
                    "epoch": int(model.current_epoch),
                    "when": when,
                    "sample_index": int(i),
                    "score_threshold": float(self.score_threshold),
                    "prompts": prompt_texts
                }
                
                torch.save(result_data, result_path)
                
                if self.debug:
                    logger.info(f"Saved evaluation result to {os.path.abspath(result_path)}")
                
                num_predictions = len(evaluation_result.predictions) if evaluation_result.predictions else 0
                if self.debug:
                    logger.info(f"Epoch {model.current_epoch} ({when}) - Sample {i}: Generated {num_predictions} predictions with threshold {self.score_threshold}")
                
                # NOTE: if no predictions at original threshold, try with lower threshold for debugging
                if num_predictions == 0 and self.debug:
                    logger.info(f"No predictions found with threshold {self.score_threshold}, trying with threshold 0.1 for debugging...")
                    debug_result = model.evaluate(
                        motion=motion_tensor,
                        prompts=prompt_texts,
                        score_threshold=0.1,
                    )
                    debug_predictions = len(debug_result.predictions) if debug_result.predictions else 0
                    logger.info(f"With threshold 0.1: Generated {debug_predictions} predictions")
                    
                    # if debug_predictions > 0:
                    #     debug_scores = [pred[3] for pred in debug_result.predictions]
                    #     logger.info(f"Debug prediction scores: {debug_scores}")
                    #     logger.info(f"Max score: {max(debug_scores):.4f}, Min score: {min(debug_scores):.4f}")
                
                if num_predictions > 0 and self.debug:
                    scores = [pred[3] for pred in evaluation_result.predictions]
                    logger.info(f"Epoch {model.current_epoch} ({when}) - Sample {i}: Prediction scores: {scores}")
                
                prediction_figure = plot_evaluation_results(
                    evaluation_result,
                    title=f"Epoch {model.current_epoch} ({when}) - Sample {i}",
                )
                if prediction_figure:
                    filename = f"epoch_{model.current_epoch}_{when}_sample_{i}_prediction.html"
                    output_path = os.path.join(self.dirpath, filename)
                    prediction_figure.write_html(output_path)
                    if self.debug:
                        logger.info(f"Saved visualization to {os.path.abspath(output_path)}")
                else:
                    if self.debug:
                        logger.warning(f"No predictions to visualize for sample {i} in epoch {model.current_epoch} ({when}). Score threshold: {self.score_threshold}")
                
                groundtruth_spans = [(p[0], s[0], s[1], 1.0) for p in raw_batch.prompts[i] for s in p[1]]
                groundtruth_result = EvaluationResult(motion_length=motion_length, predictions=groundtruth_spans)
                groundtruth_figure = plot_evaluation_results(
                    groundtruth_result,
                    title=f"Epoch {model.current_epoch} ({when}) - Sample {i} (Ground Truth)",
                )
                if groundtruth_figure:
                    gt_filename = f"epoch_{model.current_epoch}_{when}_sample_{i}_groundtruth.html"
                    gt_output_path = os.path.join(self.dirpath, gt_filename)
                    groundtruth_figure.write_html(gt_output_path)
                    if self.debug:
                        logger.info(f"Saved ground truth visualization to {os.path.abspath(gt_output_path)}")
                else:
                    if self.debug:
                        logger.warning(f"No ground truth spans to visualize for sample {i} in epoch {model.current_epoch} ({when}).")
                    
        model.train(original_mode)
        
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.sanity_checking:
            return
        if self.visualize_on_start:
            self._run_and_log_visualizations(trainer, pl_module, when="start")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.sanity_checking:
            return
        if self.visualize_on_end:
            self._run_and_log_visualizations(trainer, pl_module, when="end")