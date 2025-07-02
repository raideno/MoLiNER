# In a file like: src/model/span_representation/base.py

import torch
import typing
from abc import ABC, abstractmethod

class BaseSpanRepresentationLayer(torch.nn.Module, ABC):
    """
    Abstract base class for modules that create the final span representation.

    The role of this layer is to take a batch of fixed-size aggregated span
    vectors and transform them into their final representation, which will
    be used for scoring against prompt representations.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        aggregated_spans: torch.Tensor,
        spans_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        """
        Transforms aggregated span vectors into their final representation.

        Args:
            aggregated_spans (torch.Tensor): A single, padded tensor from a span
                aggregator.
                Shape: (batch_size, max_spans_in_batch, aggregated_embedding_dim)
            spans_masks (torch.Tensor): A padded boolean mask tensor indicating which
                spans are valid (True) vs. padding (False). This is crucial for
                ensuring operations are not performed on padding.
                Shape: (batch_size, max_spans_in_batch)

        Returns:
            torch.Tensor: A tensor containing the final representation for every
                span in the batch.
                Shape: (batch_size, max_spans_in_batch, span_representation_dim)
        """
        pass