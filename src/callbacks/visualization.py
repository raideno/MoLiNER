import os
import wandb
import torch
import typing
import typing
import logging
import traceback
import pytorch_lightning

import pytorch_lightning.loggers
import pytorch_lightning.callbacks

from src.models import MoLiNER
from src.types import RawBatch, EvaluationResult

from src.visualizations import plot_evaluation_results

logger = logging.getLogger(__name__)

THRESHOLDS = [0.25, 0.50, 0.75]

class VisualizationCallback(pytorch_lightning.callbacks.Callback):
    """
    Simple callback to generate and log visualizations of model predictions on a fixed validation batch.
    """
    def __init__(
        self,
        dirpath: str,
        batch_index: int,
        debug: bool = False,
    ):
        """
        Args:
            dirpath: Directory to save visualizations
            batch_index: Index of validation batch to visualize
            debug: Whether to output debug information
        """
        super().__init__()
        
        self.dirpath: str = dirpath
        self.batch_index: int = batch_index
        self.debug: bool = debug
        self.visualization_batch: typing.Optional[RawBatch] = None
        
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_start(self, trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule):
        if self.debug:
            logger.info(f"Setting up visualization: fetching validation batch index {self.batch_index}")
        
        validation_dataloaders = trainer.val_dataloaders
        if not validation_dataloaders:
            logger.warning("No validation dataloader found. Cannot perform visualizations.")
            return

        validation_dataloader = (
            validation_dataloaders[0] 
            if isinstance(validation_dataloaders, list) 
            else validation_dataloaders
        )
        
        if len(validation_dataloader) <= self.batch_index:
            logger.warning(
                f"Batch index {self.batch_index} out of bounds for dataloader "
                f"(size: {len(validation_dataloader)}). Disabling visualizations."
            )
            return

        for i, batch in enumerate(validation_dataloader):
            if i == self.batch_index:
                self.visualization_batch = batch
                if self.debug:
                    logger.info(f"Stored validation batch {i} with {len(batch.sid)} samples")
                break

    def on_train_epoch_end(self, trainer: pytorch_lightning.Trainer, pl_module: pytorch_lightning.LightningModule):
        if trainer.sanity_checking or self.visualization_batch is None:
            return
            
        if not isinstance(pl_module, MoLiNER):
            logger.error(f"Expected MoLiNER model but got {type(pl_module)}")
            return

        if self.debug:
            logger.info(f"Generating visualizations for epoch {pl_module.current_epoch}")

        self._generate_and_save_visualizations(pl_module, trainer)

    def _generate_and_save_visualizations(self, model: MoLiNER, trainer: pytorch_lightning.Trainer):
        original_mode = model.training
        
        model.eval()
        
        try:
            with torch.no_grad():
                threshold_results = {}
                if self.visualization_batch is not None:
                    batch = self.visualization_batch.to(model.device)
                else:
                    logger.warning("No visualization batch set. Skipping visualizations.")
                    return
                
                for threshold in THRESHOLDS:
                    threshold_results[threshold] = model.predict(batch=batch, threshold=threshold)
                
                epoch = model.current_epoch
                epoch_dir = os.path.join(self.dirpath, f"epoch_{epoch:03d}")
                os.makedirs(epoch_dir, exist_ok=True)
                
                for sample_idx in range(len(batch.sid)):
                    self._create_sample_visualization(
                        sample_idx=sample_idx,
                        batch=batch,
                        threshold_results=threshold_results,
                        epoch=epoch,
                        epoch_dir=epoch_dir,
                        trainer=trainer
                    )
        
        finally:
            model.train(original_mode)

    def _create_sample_visualization(
        self,
        sample_idx: int,
        batch: RawBatch,
        threshold_results: typing.Dict[float, typing.Any],
        epoch: int,
        epoch_dir: str,
        trainer: pytorch_lightning.Trainer
    ):
        try:
            motion_length = int(batch.motion_mask[sample_idx].sum())
            raw_prompts = batch.prompts[sample_idx]
            
            groundtruth_spans = []
            for prompt_text, spans, _ in raw_prompts:
                if spans:
                    span_list_with_scores = [(start, end, 1.0) for start, end in spans]
                    groundtruth_spans.append((f"GT: {prompt_text}", span_list_with_scores))
                else:
                    groundtruth_spans.append((f"GT: {prompt_text}", []))
            
            all_predicted_spans = []
            for threshold in THRESHOLDS:
                sample_predictions = threshold_results[threshold].predictions[sample_idx]
                
                prompt_predictions_dict = {}
                for prompt_text, span_list in sample_predictions:
                    pred_key = f"PRED({threshold}): {prompt_text}"
                    if pred_key not in prompt_predictions_dict:
                        prompt_predictions_dict[pred_key] = []
                    prompt_predictions_dict[pred_key].extend(span_list)
                
                # for pred_key, span_list in prompt_predictions_dict.items():
                #     all_predicted_spans.append((pred_key, span_list))
                
                # for prompt_text, _, _ in raw_prompts:
                #     pred_key = f"PRED({threshold}): {prompt_text}"
                #     if pred_key not in prompt_predictions_dict:
                #         all_predicted_spans.append((pred_key, []))
                
                for prompt_text, _, _ in raw_prompts:
                    pred_key = f"PRED({threshold}): {prompt_text}"
                    if pred_key in prompt_predictions_dict:
                        # NOTE: add the predictions we found
                        all_predicted_spans.append((pred_key, prompt_predictions_dict[pred_key]))
                    else:
                        # NOTE: add empty predictions for prompts with no predictions above threshold
                        all_predicted_spans.append((pred_key, []))
            
            combined_predictions = groundtruth_spans + all_predicted_spans
            
            combined_result = EvaluationResult(
                motion_length=[motion_length],
                predictions=[combined_predictions]
            )
            
            all_gt_prompts = [f"GT: {p[0]}" for p in raw_prompts]
            all_pred_prompts = []
            for threshold in THRESHOLDS:
                threshold_pred_prompts = [f"PRED({threshold}): {p[0]}" for p in raw_prompts]
                all_pred_prompts.extend(threshold_pred_prompts)
            
            all_prompts_to_show = all_gt_prompts + all_pred_prompts
            
            if self.debug:
                num_predictions = len(threshold_results[THRESHOLDS[0]].predictions[sample_idx]) if sample_idx < len(threshold_results[THRESHOLDS[0]].predictions) else 0
                logger.info(f"Sample {sample_idx}: {num_predictions} predictions, motion_length: {motion_length}")
            
            title = f"Epoch {epoch} - Sample {sample_idx} - Combined GT & All Threshold Predictions"
            filename = f"sample_{sample_idx}.html"
            output_path = os.path.join(epoch_dir, filename)
            
            figures = plot_evaluation_results(
                combined_result,
                title=title,
                all_prompts=all_prompts_to_show,
            )
            
            if figures and figures[0] is not None:
                figures[0].write_html(output_path)
                
                if self.debug:
                    logger.info(f"Saved visualization to {filename}")
                
                wandb_logger = self._get_wandb_logger(trainer)
                if wandb_logger:
                    try:
                        wandb_logger.log_table(
                            key="visualizations",
                            data=[[f"epoch_{epoch:03d}/sample_{sample_idx}", wandb.Html(output_path)]],
                            columns=["key", "visualization"]
                        )
                    except Exception as e:
                        if self.debug:
                            logger.warning(f"Failed to log to WandB: {e}")
            else:
                logger.warning(f"No figure generated for sample {sample_idx}")
                
        except Exception as exception:
            logger.error(f"Error creating visualization for sample {sample_idx}: {exception}")
            if self.debug:
                traceback.print_exc()

    def _get_wandb_logger(self, trainer: pytorch_lightning.Trainer) -> typing.Optional[pytorch_lightning.loggers.WandbLogger]:
        if trainer.logger is None:
            return None
            
        if hasattr(trainer, 'loggers') and trainer.loggers:
            for logger_instance in trainer.loggers:
                if hasattr(logger_instance, '__class__') and 'WandbLogger' in str(logger_instance.__class__):
                    # type: ignore
                    return logger_instance
        elif hasattr(trainer.logger, '__class__') and 'WandbLogger' in str(trainer.logger.__class__):
            # type: ignore
            return trainer.logger
            
        return None
