import os
import torch
import logging
import typing

import numpy as np
import pytorch_lightning as pl

from src.model import MoLiNER
from pytorch_lightning.callbacks import Callback
from src.types import RawBatch, ProcessedBatch, EvaluationResult
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
        score_threshold: typing.Union[float, typing.List[float]] = 0.5,
        visualize_on_start: bool = True,
        visualize_on_end: bool = True,
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
            visualize_on_start (bool): Whether to run visualization at the start of the epoch.
            visualize_on_end (bool): Whether to run visualization at the end of the epoch.
            visualization_frequency (int): Run visualization every N epochs (default: 1).
            skip_html_generation (bool): Skip HTML generation for faster execution (default: False).
        """
        super().__init__()
        
        self.dirpath = dirpath
        self.batch_index = batch_index
        self.num_samples = num_samples
        if isinstance(score_threshold, (int, float)):
            self.score_thresholds = [float(score_threshold)]
        else:
            self.score_thresholds = [float(t) for t in score_threshold]
        self.visualize_on_start = visualize_on_start
        self.visualize_on_end = visualize_on_end
        self.visualization_batch: typing.Optional["RawBatch"] = None
        self.debug = debug
        self.visualization_frequency = max(1, visualization_frequency)
        self.skip_html_generation = skip_html_generation
        
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
                
    def _move_batch_to_cpu(self, batch: "RawBatch") -> "RawBatch":
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
            logger.info(f"Generating visualizations for epoch {model.current_epoch} ({when})...")
        
        original_mode = model.training
        
        model.eval()
        
        all_results = {}
        
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
                    logger.info(f"Epoch {model.current_epoch} ({when}) - Sample {i}: Motion tensor shape: {motion_tensor.shape}")
                    logger.info(f"Epoch {model.current_epoch} ({when}) - Sample {i}: Prompts: {prompt_texts[:3]}...")
                
                formatted_prompts = [(text, [], True) for text in prompt_texts]
                
                eval_raw_batch = RawBatch(
                    sid=[0],
                    dataset_name=["evaluation"],
                    amass_relative_path=["none"],
                    raw_motion=torch.zeros_like(motion_tensor.unsqueeze(0)),
                    transformed_motion=motion_tensor.unsqueeze(0).to(model.device),
                    motion_mask=torch.ones(1, motion_tensor.shape[0], dtype=torch.bool).to(model.device),
                    prompts=[formatted_prompts]
                )
                
                eval_processed_batch = ProcessedBatch.from_raw_batch(
                    raw_batch=eval_raw_batch,
                    encoder=model.prompts_tokens_encoder
                )
                
                forward_output = model.forward(
                    eval_processed_batch,
                    batch_index=0
                )
                
                threshold_results = {}
                for threshold in self.score_thresholds:
                    decoded_results = model.decoder.decode(
                        forward_output=forward_output,
                        prompts=prompt_texts,
                        score_threshold=threshold,
                    )
                    evaluation_result = decoded_results[0]
                    threshold_results[threshold] = evaluation_result
                
                primary_threshold = self.score_thresholds[0]
                evaluation_result = threshold_results[primary_threshold]
                
                result_data = {
                    "motion_length": int(evaluation_result.motion_length),
                    "num_predictions": len(evaluation_result.predictions) if evaluation_result.predictions else 0,
                    "epoch": int(model.current_epoch),
                    "when": when,
                    "sample_index": int(i),
                    "score_threshold": float(primary_threshold),
                    "num_prompts": len(prompt_texts)
                }
                
                all_results[i] = {
                    'result_data': result_data,
                    'threshold_results': threshold_results,
                    'motion_length': motion_length,
                    'raw_prompts': raw_batch.prompts[i]
                }
                
                num_predictions = len(evaluation_result.predictions) if evaluation_result.predictions else 0
                if self.debug:
                    logger.info(f"Epoch {model.current_epoch} ({when}) - Sample {i}: Generated {num_predictions} predictions with threshold {primary_threshold}")
                
                if num_predictions == 0 and self.debug and primary_threshold > 0.1:
                    debug_decoded_results = model.decoder.decode(
                        forward_output=forward_output,
                        prompts=prompt_texts,
                        score_threshold=0.1,
                    )
                    debug_result = debug_decoded_results[0]
                    debug_predictions = len(debug_result.predictions) if debug_result.predictions else 0
                    logger.info(f"Debug: With threshold 0.1: Generated {debug_predictions} predictions")
        
        model.train(original_mode)
        
        if not self.skip_html_generation:
            self._save_visualizations_async(all_results, model.current_epoch, when, trainer)
        else:
            if self.debug:
                logger.info(f"Skipping HTML generation for faster execution")
                
    def _save_visualizations_async(self, all_results: dict, epoch: int, when: str, trainer: pl.Trainer):
        wandb_logger = None
        if trainer.logger is not None:
            if hasattr(trainer, 'loggers') and trainer.loggers:
                for logger_instance in trainer.loggers:
                    if hasattr(logger_instance, '__class__') and 'WandBLogger' in str(logger_instance.__class__):
                        wandb_logger = logger_instance
                        break
            elif hasattr(trainer.logger, '__class__') and 'WandBLogger' in str(trainer.logger.__class__):
                wandb_logger = trainer.logger
        
        epoch_dir = os.path.join(self.dirpath, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        for i, result_info in all_results.items():
            try:
                result_filename = f"{when}_sample_{i}_evaluation_result.pt"
                result_path = os.path.join(epoch_dir, result_filename)
                torch.save(result_info['result_data'], result_path)
                
                motion_length = result_info['motion_length']
                groundtruth_spans = [(f"GT: {p[0]}", s[0], s[1], 1.0) for p in result_info['raw_prompts'] for s in p[1]]
                
                for threshold, threshold_evaluation_result in result_info['threshold_results'].items():
                    # NOTE: prefix predicted spans with "PRED: "
                    predicted_spans = [(f"PRED: {pred[0]}", pred[1], pred[2], pred[3]) for pred in threshold_evaluation_result.predictions]
                    
                    combined_predictions = groundtruth_spans + predicted_spans
                    combined_result = EvaluationResult(motion_length=motion_length, predictions=combined_predictions)
                    
                    all_gt_prompts = [f"GT: {p[0]}" for p in result_info['raw_prompts']]
                    all_pred_prompts = [f"PRED: {p[0]}" for p in result_info['raw_prompts']]
                    all_prompts_to_show = all_gt_prompts + all_pred_prompts
                    
                    combined_figure = plot_evaluation_results(
                        combined_result,
                        title=f"Epoch {epoch} ({when}) - Sample {i} (Threshold: {threshold}) - Combined GT & Predictions",
                        all_prompts=all_prompts_to_show,
                    )
                    if combined_figure:
                        filename = f"{when}_sample_{i}_combined_thresh_{threshold:.3f}.html"
                        output_path = os.path.join(epoch_dir, filename)
                        combined_figure.write_html(output_path)
                        if self.debug:
                            logger.info(f"Saved combined visualization to epoch_{epoch:03d}/{os.path.basename(output_path)}")
                        
                        if wandb_logger is not None:
                            try:
                                # type: ignore
                                wandb_logger.log_html_visualization(
                                    html_path=output_path,
                                    key=f"visualizations/epoch_{epoch:03d}/{when}_sample_{i}_threshold_{threshold:.3f}_combined"
                                )
                            except Exception as e:
                                if self.debug:
                                    logger.warning(f"Failed to log combined visualization to WandB: {e}")
                    
            except Exception as e:
                logger.error(f"Error saving visualization for sample {i}: {e}")
                continue
        
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