import torch
import typing

from .index import BaseSpanFramesAggregator

class ConcatFirstLastAggregator(BaseSpanFramesAggregator):
    """
    A simple span aggregator that creates a span representation by
    concatenating the feature vectors of its first and last frames.
    
    This implementation is fully vectorized using torch.gather.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        span_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        For each span defined in `span_indices`, it gathers the first and last
        frame features from `motion_features` and concatenates them.
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