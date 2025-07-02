import torch
import typing

from abc import ABC, abstractmethod

class BaseSpanFramesAggregator(torch.nn.Module, ABC):
    """
    Abstract base class for all span frame aggregator modules.

    The role of an aggregator is to use generated span indices to gather
    frame features from the original motion sequence and produce a single,
    fixed-dimension representation for each span.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        span_mask: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        """
        Aggregates frames within each span to produce a fixed-size representation.

        Args:
            motion_features (torch.Tensor): The original motion frame embeddings.
                Shape: (batch_size, seq_len, embed_dim)
            span_indices (torch.Tensor): A tensor of [start, end] frame indices for all
                generated spans in the batch. Padded indices are typically -1.
                Shape: (batch_size, max_spans, 2)
            span_mask (torch.Tensor): A boolean tensor indicating which spans are valid (True)
                vs. padding (False).
                Shape: (batch_size, max_spans)

        Returns:
            torch.Tensor: A single, padded tensor containing the aggregated representation
                for every span in the batch.
                Shape: (batch_size, max_spans, aggregated_embedding_dim)
        """
        pass