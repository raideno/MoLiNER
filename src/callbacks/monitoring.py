import torch
import typing
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from src.types import ForwardOutput, ProcessedBatch
from src.model.losses.helpers import create_target_matrix, create_loss_mask


class MonitoringCallback(Callback):
    """
    Callback for monitoring target matrix statistics during training and validation.
    This callback automatically computes and logs statistics about the target matrix to help monitor training progress.
    """
    
    def __init__(self):
        super().__init__()
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: typing.Union[typing.Mapping[str, typing.Any], torch.Tensor, None],
        batch: typing.Any,
        batch_idx: int,
    ) -> None:
        if outputs is not None and isinstance(outputs, dict):
            if "forward_output" in outputs and "processed_batch" in outputs:
                target_matrix_stats = self.compute_stats(
                    outputs["forward_output"], 
                    outputs["processed_batch"]
                )
                if target_matrix_stats:
                    self._log_stats(
                        stats=target_matrix_stats,
                        lightning_module=pl_module,
                        prefix="train",
                        batch_size=outputs.get("batch_size", 1)
                    )
    
    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: typing.Union[typing.Mapping[str, typing.Any], torch.Tensor, None],
        batch: typing.Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is not None and isinstance(outputs, dict):
            if "forward_output" in outputs and "processed_batch" in outputs:
                target_matrix_stats = self.compute_stats(
                    outputs["forward_output"], 
                    outputs["processed_batch"]
                )
                if target_matrix_stats:
                    self._log_stats(
                        stats=target_matrix_stats,
                        lightning_module=pl_module,
                        prefix="val",
                        batch_size=outputs.get("batch_size", 1)
                    )
    
    def compute_stats(
        self,
        forward_output: ForwardOutput,
        batch: ProcessedBatch,
    ) -> typing.Dict[str, float]:
        """
        Compute statistics for monitoring training progress.
        
        Args:
            forward_output: Output from model forward pass
            batch: Processed batch data
            
        Returns:
            Dict containing various statistics about the target matrix
        """
        if batch.target_spans is None:
            return {}
        
        target_matrix, unmatched = create_target_matrix(forward_output, batch, 1.0)
        loss_mask = create_loss_mask(forward_output, batch)
        
        return self._compute_target_matrix_stats(
            target_matrix=target_matrix,
            similarity_matrix=forward_output.similarity_matrix,
            loss_mask=loss_mask,
            unmatched=unmatched
        )
    
    def _compute_target_matrix_stats(
        self,
        target_matrix: torch.Tensor,
        similarity_matrix: torch.Tensor,
        loss_mask: torch.Tensor,
        unmatched: int
    ) -> typing.Dict[str, float]:
        """
        Compute statistics for the target matrix to monitor training progress.
        
        Args:
            target_matrix: Binary target matrix (batch_size, num_prompts, num_spans)
            similarity_matrix: Predicted similarity matrix (batch_size, num_prompts, num_spans)
            loss_mask: Mask indicating valid (non-padding) pairs
            unmatched: Number of unmatched spans in the batch
            
        Returns:
            Dict containing various statistics about the target matrix
        """
        total_valid_pairs = loss_mask.sum().item()
        
        valid_targets = target_matrix * loss_mask
        valid_predictions = similarity_matrix * loss_mask
        
        if total_valid_pairs == 0:
            return {
                "pred_min": 0.0,
                "pred_max": 0.0,
                "pred_mean": 0.0,
                "unmatched": unmatched
            }
        
        valid_predictions = torch.sigmoid(valid_predictions)
        
        pred_min = valid_predictions.min().item()
        pred_max = valid_predictions.max().item()
        pred_mean = valid_predictions.sum().item() / total_valid_pairs
        
        return {
            "pred_min": pred_min,
            "pred_max": pred_max,
            "pred_mean": pred_mean,
            "unmatched": unmatched
        }
    
    def _log_stats(
        self,
        stats: typing.Dict[str, float],
        lightning_module: pl.LightningModule,
        prefix: str,
        batch_size: int,
        on_step: bool = True,
        on_epoch: bool = True,
    ) -> None:
        if not stats:
            return
        
        for key, value in stats.items():
            log_key = f"{prefix}/{key}"
            lightning_module.log(
                log_key,
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                batch_size=batch_size,
                prog_bar=False
            )
