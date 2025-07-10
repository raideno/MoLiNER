import torch
import typing

from ._base import BaseSpanRepresentationLayer

class MaxPoolingSpanRepresentationLayer(BaseSpanRepresentationLayer):
    """
    NOTE: A span representation layer that aggregates span frames by taking the
    element-wise maximum of their embeddings.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        spans_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, max_spans, _ = span_indices.shape
        embedding_dimension = motion_features.shape[-1]
        device = motion_features.device

        span_representations = torch.zeros(batch_size, max_spans, embedding_dimension, device=device)

        for i in range(batch_size):
            for j in range(max_spans):
                if spans_masks[i, j]:
                    start, end = span_indices[i, j]
                    # NOTE: end is inclusive, so we need to add 1
                    span_frames = motion_features[i, start:end + 1]
                    if span_frames.shape[0] > 0:
                        span_representations[i, j] = torch.max(span_frames, dim=0)[0]

        return span_representations

class MeanPoolingSpanRepresentationLayer(BaseSpanRepresentationLayer):
    """
    NOTE: A span representation layer that aggregates span frames by taking the
    element-wise mean of their embeddings.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        spans_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, max_spans, _ = span_indices.shape
        embedding_dimension = motion_features.shape[-1]
        device = motion_features.device

        span_representations = torch.zeros(batch_size, max_spans, embedding_dimension, device=device)

        for i in range(batch_size):
            for j in range(max_spans):
                if spans_masks[i, j]:
                    start, end = span_indices[i, j]
                    # NOTE: end is inclusive, so we need to add 1
                    span_frames = motion_features[i, start:end + 1]
                    if span_frames.shape[0] > 0:
                        span_representations[i, j] = torch.mean(span_frames, dim=0)

        return span_representations


class MinPoolingSpanRepresentationLayer(BaseSpanRepresentationLayer):
    """
    NOTE: A span representation layer that aggregates span frames by taking the
    element-wise minimum of their embeddings.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        spans_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, max_spans, _ = span_indices.shape
        embedding_dimension = motion_features.shape[-1]
        device = motion_features.device

        span_representations = torch.zeros(batch_size, max_spans, embedding_dimension, device=device)

        for i in range(batch_size):
            for j in range(max_spans):
                if spans_masks[i, j]:
                    start, end = span_indices[i, j]
                    # NOTE: end is inclusive, so we need to add 1
                    span_frames = motion_features[i, start:end + 1]
                    if span_frames.shape[0] > 0:
                        span_representations[i, j] = torch.min(span_frames, dim=0)[0]

        return span_representations
