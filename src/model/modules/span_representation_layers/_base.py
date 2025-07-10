import torch
import typing

from abc import ABC, abstractmethod

class BaseSpanRepresentationLayer(torch.nn.Module, ABC):
    """
    Abstract base class for modules that create the final span representation.

    The role of this layer is to take motion frame embeddings and span indices,
    aggregate the frames within each span, and transform them into their final 
    representation, which will be used for scoring against prompt representations.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        spans_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        """
        Aggregates frames within each span and transforms them into their final representation.

        Args:
            motion_features (torch.Tensor): The original motion frame embeddings.
                Shape: (batch_size, seq_len, embed_dim)
            span_indices (torch.Tensor): A tensor of [start, end] frame indices for all
                generated spans in the batch. Padded indices are typically -1.
                Shape: (batch_size, max_spans, 2)
            spans_masks (torch.Tensor): A boolean tensor indicating which spans are valid (True)
                vs. padding (False).
                Shape: (batch_size, max_spans)

        Returns:
            torch.Tensor: A tensor containing the final representation for every
                span in the batch.
                Shape: (batch_size, max_spans, span_representation_dim)
        """
        pass
