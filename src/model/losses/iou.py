import torch
import typing

from src.types import ForwardOutput, ProcessedBatch

from ._base import BaseLoss

from .helpers import (
    create_iou_target_matrix,
    create_loss_mask,
    create_negatives_mask,
    reduce
)

class IoULoss(BaseLoss):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "sum",
        label_smoothing: float = 0.0,
        negatives_type: str = "labels",
        negatives_probability: float = 1.0,
        ignore_index: int = -100,
        iou_threshold: float = 0.5,
    ):
        """
        Initialize IoU-based Loss that uses IoU scores as continuous targets instead of binary labels.
        
        Args:
            alpha (float): Weighting factor for positive examples (0-1) - used with focal loss
            gamma (float): Focusing parameter to down-weight easy examples - used with focal loss
            reduction (str): Reduction method ('none', 'mean', 'sum')
            label_smoothing (float): Label smoothing factor (0-1)
            negatives_type (str): Type of negative sampling ('labels', 'global', 'span')
            negatives_probability (float): Probability of keeping negative examples
            ignore_index (int): Index to ignore in loss computation
            iou_threshold (float): IoU threshold for considering a match (for counting purposes)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        self.negatives_type = negatives_type
        self.negatives_probability = negatives_probability
        
        self.ignore_index = ignore_index
        self.iou_threshold = iou_threshold
    
    def _compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss based on the specified loss type.
        
        Args:
            inputs: Predicted similarity scores (logits)
            targets: IoU scores as continuous targets [0, 1]
        
        Returns:
            torch.Tensor: Loss values (unreduced)
        """
        # Apply label smoothing if needed
        if self.label_smoothing != 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Create a mask to ignore specified index
        valid_mask = targets != self.ignore_index
        
        # Apply focal loss with continuous targets
        pred_probs = torch.sigmoid(inputs)
        
        # Compute BCE loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        
        # Apply focal loss modulation if gamma > 0
        if self.gamma > 0:
            # For continuous targets, use the IoU score as the certainty
            p_t = pred_probs * targets + (1 - pred_probs) * (1 - targets)
            loss = loss * ((1 - p_t) ** self.gamma)
        
        # Apply alpha weighting if specified
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        # Apply the valid mask to the loss
        loss = loss * valid_mask
        
        return loss
    
    def forward(
        self,
        forward_output: ForwardOutput,
        batch: ProcessedBatch,
        batch_index: typing.Optional[int] = None,
    ) -> typing.Tuple[torch.Tensor, int]:
        if batch.target_spans is None:
            raise ValueError("Cannot compute loss without target spans (training data)")
        
        predicted_logits = forward_output.similarity_matrix
        
        # Use IoU-based target matrix instead of exact matching
        target_iou_scores, unmatched_spans_count = create_iou_target_matrix(
            forward_output, batch, iou_threshold=self.iou_threshold
        )
        
        # NOTE: (batch, prompts, spans)
        # Indicates which pairs are not padding and should be considered for loss computation
        loss_mask = create_loss_mask(forward_output, batch)
        
        all_losses = self._compute_loss(
            inputs=predicted_logits,
            targets=target_iou_scores,
        )
        
        # NOTE: ignore padding
        all_losses = all_losses * loss_mask
        
        negatives_mask = create_negatives_mask(
            logits=target_iou_scores,
            type=self.negatives_type,
            negatives=self.negatives_probability
        )
        
        all_losses = all_losses * negatives_mask
        
        loss = reduce(
            logits=all_losses,
            reduction=self.reduction
        )
        
        return loss, unmatched_spans_count
