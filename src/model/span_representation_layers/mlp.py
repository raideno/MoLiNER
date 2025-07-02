import torch
import torch.nn as nn

from .index import BaseSpanRepresentationLayer

class MLPSpanRepresentationLayer(BaseSpanRepresentationLayer):
    """
    A span representation layer that uses a two-layer MLP to transform
    aggregated span vectors into their final representation.
    """
    def __init__(self, input_dim: int, representation_dim: int, dropout_rate: float = 0.1):
        """
        Initializes the MLPSpanRepresentationLayer.

        Args:
            input_dim (int): The dimension of the input aggregated span vectors.
                             For ConcatFirstLastAggregator, this is 2 * embed_dim.
            representation_dim (int): The dimension of the final output representation.
            dropout_rate (float): The dropout rate to apply for regularization.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, representation_dim)
        )

    def forward(
        self,
        aggregated_spans: torch.Tensor,
        spans_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Passes the aggregated spans through the MLP and applies a mask to zero
        out representations for padded spans.
        """
        # (NOTE: batch_size, max_spans, representation_dim)
        representations = self.mlp(aggregated_spans)
        
        # NOTE: zero out padded spans
        masked_representations = representations * spans_masks.unsqueeze(-1)
        
        return masked_representations