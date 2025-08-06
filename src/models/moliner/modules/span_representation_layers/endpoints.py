import torch
import typing

from src.models.helpers import create_projection_layer

from ._base import BaseSpanRepresentationLayer

class EndpointsSpanRepresentationLayer(BaseSpanRepresentationLayer):
    """
    A span representation layer that aggregates span frames by concatenating
    the first and last frame embeddings, then uses an MLP to transform
    them into their final representation.
    """
    def __init__(self, motion_embed_dim: int, representation_dim: int, dropout: float):
        """
        Args:
            motion_embed_dim (int): The dimension of the motion frame embeddings.
            representation_dim (int): The dimension of the final output representation.
            dropout (float): The dropout rate to apply for regularization.
        """
        super().__init__()
        
        self.start_proj = create_projection_layer(motion_embed_dim, dropout)
        self.end_proj = create_projection_layer(motion_embed_dim, dropout)
        
        # NOTE: operates on the concatenation of first and last frame embeddings
        self.out_project = create_projection_layer(motion_embed_dim * 2, dropout, representation_dim)

    def forward(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        spans_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        first_frames, last_frames = self._get_span_endpoints(
            motion_features, span_indices
        )

        first_frames = self.start_proj(first_frames)
        last_frames = self.end_proj(last_frames)

        aggregated_spans = torch.cat((first_frames, last_frames), dim=-1)
        
        aggregated_spans.mul_(spans_masks.unsqueeze(-1))
        # TODO: check if this is necessary and if it is badly impacting the model's performance
        # aggregated_spans = aggregated_spans * spans_masks.unsqueeze(-1)
        
        representations = self.out_project(aggregated_spans)
        
        representations.mul_(spans_masks.unsqueeze(-1))
        # representations = representations * spans_masks.unsqueeze(-1)
        
        return representations
    
    def _get_span_endpoints(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_spans, _ = span_indices.shape
        _, _, embed_dim = motion_features.shape

        if batch_size == 0 or max_spans == 0:
            empty_tensor = torch.empty(batch_size, max_spans, embed_dim, device=motion_features.device)
            return empty_tensor, empty_tensor

        start_indices = span_indices[..., 0]
        end_indices = span_indices[..., 1]

        # NOTE: padded spans might have start / end set to -1
        start_indices_clamped = start_indices.clamp(min=0)
        end_indices_clamped = end_indices.clamp(min=0)

        start_gather_idx = start_indices_clamped.unsqueeze(-1).expand(-1, -1, embed_dim)
        end_gather_idx = end_indices_clamped.unsqueeze(-1).expand(-1, -1, embed_dim)

        first_frames = torch.gather(motion_features, 1, start_gather_idx)
        last_frames = torch.gather(motion_features, 1, end_gather_idx)
                
        return first_frames, last_frames