import torch
import typing

from src.types import ForwardOutput, ProcessedBatch
from src.model.losses.helpers import create_target_matrix, create_loss_mask

class Monitoring:
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def compute_stats(
        self,
        forward_output: ForwardOutput,
        batch: ProcessedBatch,
    ) -> typing.Dict[str, float]:
        if batch.target_spans is None:
            return {}
        
        target_matrix, _ = create_target_matrix(forward_output, batch)
        loss_mask = create_loss_mask(forward_output, batch)
        
        return self._compute_target_matrix_stats(
            target_matrix=target_matrix,
            similarity_matrix=forward_output.similarity_matrix,
            loss_mask=loss_mask,
            threshold=self.threshold
        )
    
    def _compute_target_matrix_stats(
        self,
        target_matrix: torch.Tensor,
        similarity_matrix: torch.Tensor,
        loss_mask: torch.Tensor,
        threshold: float,
    ) -> typing.Dict[str, float]:
        """
        Compute statistics for the target matrix to monitor training progress.
        
        Args:
            target_matrix: Binary target matrix (batch_size, num_prompts, num_spans)
            similarity_matrix: Predicted similarity matrix (batch_size, num_prompts, num_spans)
            loss_mask: Mask indicating valid (non-padding) pairs
            threshold: Threshold for counting predictions above threshold
            
        Returns:
            Dict containing various statistics about the target matrix
        """
        valid_targets = target_matrix * loss_mask
        valid_predictions = similarity_matrix * loss_mask
        
        total_valid_pairs = loss_mask.sum().item()
        positive_pairs = valid_targets.sum().item()
        negative_pairs = total_valid_pairs - positive_pairs
        
        if total_valid_pairs == 0:
            return {
                "total_valid_pairs": 0.0,
                "positive_pairs": 0.0,
                "negative_pairs": 0.0,
                "positive_ratio": 0.0,
                "target_min": 0.0,
                "target_max": 0.0,
                "target_mean": 0.0,
                "pred_min": 0.0,
                "pred_max": 0.0,
                "pred_mean": 0.0,
                "pred_above_threshold": 0.0,
                "true_positives": 0.0,
                "precision_at_threshold": 0.0,
                "recall_at_threshold": 0.0,
            }
        
        target_min = valid_targets.min().item()
        target_max = valid_targets.max().item()
        target_mean = valid_targets.sum().item() / total_valid_pairs
        
        pred_min = valid_predictions.min().item()
        pred_max = valid_predictions.max().item()
        pred_mean = valid_predictions.sum().item() / total_valid_pairs
        
        pred_above_threshold = (valid_predictions > threshold).sum().item()
        
        true_positives = ((valid_predictions > threshold) * valid_targets).sum().item()
        
        precision_at_threshold = true_positives / max(pred_above_threshold, 1)
        recall_at_threshold = true_positives / max(positive_pairs, 1)
        
        return {
            "total_valid_pairs": float(total_valid_pairs),
            "positive_pairs": float(positive_pairs),
            "negative_pairs": float(negative_pairs),
            "positive_ratio": positive_pairs / total_valid_pairs,
            "target_min": target_min,
            "target_max": target_max,
            "target_mean": target_mean,
            "pred_min": pred_min,
            "pred_max": pred_max,
            "pred_mean": pred_mean,
            "pred_above_threshold": float(pred_above_threshold),
            "true_positives": float(true_positives),
            "precision_at_threshold": precision_at_threshold,
            "recall_at_threshold": recall_at_threshold,
        }
    
    def log_stats(
        self,
        stats: typing.Dict[str, float],
        lightning_module,
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