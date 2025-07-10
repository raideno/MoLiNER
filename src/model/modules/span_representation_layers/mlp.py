import torch
import typing

from .index import BaseSpanRepresentationLayer

class MLPSpanRepresentationLayer(BaseSpanRepresentationLayer):
    """
    A span representation layer that aggregates span frames by concatenating
    the first and last frame embeddings, then uses a two-layer MLP to transform
    them into their final representation.
    """
    def __init__(self, motion_embed_dim: int, representation_dim: int, dropout_rate: float = 0.1):
        """
        Initializes the MLPSpanRepresentationLayer.

        Args:
            motion_embed_dim (int): The dimension of the motion frame embeddings.
                                   The aggregated input will be 2 * motion_embed_dim.
            representation_dim (int): The dimension of the final output representation.
            dropout_rate (float): The dropout rate to apply for regularization.
        """
        super().__init__()
        input_dim = 2 * motion_embed_dim  # Concatenation of first and last frames
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(input_dim, representation_dim)
        )

    def forward(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        spans_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        """
        Aggregates frames within each span using concat first-last strategy,
        then passes through MLP and applies mask to zero out representations 
        for padded spans.
        """
        # First, aggregate the spans using concat first-last strategy
        aggregated_spans = self._aggregate_spans_concat_first_last(
            motion_features, span_indices, spans_masks
        )
        
        # Then pass through MLP
        representations = self.mlp(aggregated_spans)
        
        # Zero out padded spans
        masked_representations = representations * spans_masks.unsqueeze(-1)
        
        return masked_representations
    
    def _aggregate_spans_concat_first_last(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        span_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregates spans by concatenating the feature vectors of their first and last frames.
        This implementation is fully vectorized using torch.gather.
        """
        batch_size, max_spans, _ = span_indices.shape
        _, _, embed_dim = motion_features.shape

        if batch_size == 0 or max_spans == 0:
            return torch.empty(batch_size, max_spans, 2 * embed_dim, device=motion_features.device)

        start_indices = span_indices[..., 0]
        end_indices = span_indices[..., 1]

        # NOTE: padded spans might have start / end set to -1
        start_indices_clamped = start_indices.clamp(min=0)
        end_indices_clamped = end_indices.clamp(min=0)

        start_gather_idx = start_indices_clamped.unsqueeze(-1).expand(-1, -1, embed_dim)
        end_gather_idx = end_indices_clamped.unsqueeze(-1).expand(-1, -1, embed_dim)

        first_frames = torch.gather(motion_features, 1, start_gather_idx)
        last_frames = torch.gather(motion_features, 1, end_gather_idx)

        aggregated_spans = torch.cat((first_frames, last_frames), dim=-1)

        masked_aggregated_spans = aggregated_spans * span_mask.unsqueeze(-1)
                
        return masked_aggregated_spans