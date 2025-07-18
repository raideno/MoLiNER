import torch
import typing

from src.types import ForwardOutput, ProcessedBatch

from ._base import BaseLoss

from .helpers import (
    create_target_matrix,
    create_loss_mask,
    create_negatives_mask,
    reduce
)

class FocalLoss(BaseLoss):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "sum",
        label_smoothing: float = 0.0,
        negatives_type: str = "labels",
        negatives_probability: float = 1.0,
        ignore_index: int = -100
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha (float): Weighting factor for positive examples (0-1)
            gamma (float): Focusing parameter to down-weight easy examples
            reduction (str): Reduction method ('none', 'mean', 'sum')
            label_smoothing (float): Label smoothing factor (0-1)
            negatives_type (str): Type of negative sampling ('labels', 'global', 'span')
            negatives_probability (float): Probability of keeping negative examples
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        self.negatives_type = negatives_type
        self.negatives_probability = negatives_probability
        
        self.ignore_index = ignore_index
    
    def _focal_loss_with_logits(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Focal loss values (unreduced)
        """
        # Apply label smoothing if needed
        if self.label_smoothing != 0:
            # with torch.no_grad():
            #     targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Apply sigmoid activation to inputs
        p = torch.sigmoid(inputs)
        
        # Compute the binary cross-entropy loss without reduction
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction="none"
        )
        
        # Create a mask to ignore specified index
        valid_mask = targets != self.ignore_index
        
        # Apply the valid mask to the loss
        bce_loss = bce_loss * valid_mask
        
        # Apply focal loss modulation if gamma > 0
        if self.gamma > 0:
            p_t = p * targets + (1 - p) * (1 - targets)
            bce_loss = bce_loss * ((1 - p_t) ** self.gamma)
        
        # Apply alpha weighting if specified
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss
        
        return bce_loss
    
    def forward(
        self,
        forward_output: ForwardOutput,
        batch: ProcessedBatch,
        batch_index: typing.Optional[int] = None,
    ) -> typing.Tuple[torch.Tensor, int]:
        if batch.target_spans is None:
            raise ValueError("Cannot compute loss without target spans (training data)")
        
        predicted_logits = forward_output.similarity_matrix
        
        target_logits, unmatched_spans_count = create_target_matrix(forward_output, batch)
        
        # NOTE: (batch, prompts, spans)
        # Indicates which pairs are not padding and should be considered for loss computation
        loss_mask = create_loss_mask(forward_output, batch)
        
        all_losses = self._focal_loss_with_logits(
            inputs=predicted_logits,
            targets=target_logits,
        )
        
        # NOTE: ignore padding
        all_losses = all_losses * loss_mask
        
        negatives_mask = create_negatives_mask(
            logits=target_logits,
            type=self.negatives_type,
            negatives=self.negatives_probability
        )
        
        all_losses = all_losses * negatives_mask
        
        loss = reduce(
            logits=all_losses,
            reduction=self.reduction,
            valid_mask=loss_mask
        )
        
        return loss, unmatched_spans_count
