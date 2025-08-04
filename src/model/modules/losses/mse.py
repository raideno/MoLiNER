import torch
import typing

from src.types import MolinerForwardOutput, RawBatch

from .helpers import (
    create_target_matrix,
    create_loss_mask,
    create_negatives_mask,
    reduce
)

from ._base import BaseLoss

class MSELoss(BaseLoss):
    def __init__(
        self,
        reduction: str,
        negatives_type: str,
        negatives_probability: float,
        ignore_index: int,
        threshold: float
    ):
        super().__init__()
        self.reduction = reduction
        self.negatives_type = negatives_type
        self.negatives_probability = negatives_probability
        self.ignore_index = ignore_index
        self.threshold = threshold

    def forward(
        self,
        forward_output: MolinerForwardOutput,
        batch: RawBatch,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        if batch.target_spans is None:
            raise ValueError("Cannot compute loss without target spans (training data)")
        
        predicted_logits = forward_output.similarity_matrix
        
        target_logits = create_target_matrix(
            forward_output=forward_output,
            batch=batch,
            threshold=self.threshold
        )

        loss_mask = create_loss_mask(forward_output, batch)

        negatives_mask = create_negatives_mask(
            logits=target_logits,
            type=self.negatives_type,
            negatives=self.negatives_probability
        )

        total_mask = loss_mask * negatives_mask

        raw_loss = torch.nn.functional.mse_loss(predicted_logits, target_logits, reduction="none")

        masked_loss = raw_loss * total_mask

        loss = reduce(
            logits=masked_loss,
            reduction=self.reduction,
            valid_mask=loss_mask
        )

        return loss