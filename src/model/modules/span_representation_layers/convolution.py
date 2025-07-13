import torch
import typing

from ._base import BaseSpanRepresentationLayer

class ConvolutionalSpanRepresentationLayer(BaseSpanRepresentationLayer):
    """
    A span representation layer that uses 1D convolutions to aggregate frame embeddings.
    We define one convolution layer for each possible span width, or a single shared convolution
    """

    def __init__(
        self,
        motion_embed_dim: int,
        representation_dimension: int,
        max_span_width: int,
        min_span_width: int = 1,
        shared_weights: bool = False
    ):
        super().__init__()
        
        if not isinstance(max_span_width, int) or max_span_width < 1:
            raise ValueError("max_span_width must be a positive integer.")
        if not isinstance(min_span_width, int) or min_span_width < 1:
            raise ValueError("min_span_width must be a positive integer.")
        if min_span_width > max_span_width:
            raise ValueError("min_span_width must be less than or equal to max_span_width.")
        
        self.shared_weights = shared_weights
        self.max_span_width = max_span_width
        self.min_span_width = min_span_width
        
        conv_output_dim = representation_dimension

        if shared_weights:
            # NOTE: maybe will push the model to learn features shared between spans of different sizes
            self.convolution = torch.nn.Conv1d(motion_embed_dim, conv_output_dim, kernel_size=max_span_width, padding='same')
        else:
            self.convolutions = torch.nn.ModuleList([
                torch.nn.Conv1d(motion_embed_dim, conv_output_dim, kernel_size=k) for k in range(min_span_width, max_span_width + 1)
            ])
        
        self.projection = torch.nn.Linear(conv_output_dim, representation_dimension)

    def forward(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        spans_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        # motion_features: (batch_size, seq_len, embed_dim)
        # span_indices: (batch_size, max_spans, 2)
        # spans_masks: (batch_size, max_spans)

        batch_size, max_spans, _ = span_indices.shape
        
        embedding_dimension = motion_features.shape[-1]
        representation_dimension = self.projection.out_features
        device = motion_features.device

        span_representations = torch.zeros(batch_size, max_spans, representation_dimension, device=device)

        # (batch_size, embed_dim, seq_len)
        motion_features_t = motion_features.transpose(1, 2)

        for i in range(batch_size):
            for j in range(max_spans):
                if spans_masks[i, j]:
                    start, end = span_indices[i, j]
                    # NOTE: end is inclusive
                    span_width = end - start + 1

                    if span_width > 0:
                        if self.min_span_width <= span_width <= self.max_span_width:
                            # NOTE: (1, embed_dim, span_width)
                            span_frames = motion_features_t[i:i+1, :, start:end + 1]

                            if self.shared_weights:
                                # NOTE: We might need more sophisticated padding or handling for shared weights
                                # For simplicity, we apply the conv and take the last output
                                convolution_output = self.convolution(span_frames) # (1, conv_output_dim, span_width)
                                span_rep = convolution_output[:, :, -1]
                            else:
                                if span_width <= self.max_span_width:
                                    # (1, conv_output_dim, 1)
                                    convolution_output = self.convolutions[span_width - self.min_span_width](span_frames)
                                    span_rep = convolution_output.squeeze(-1)
                        else:
                            raise ValueError(f"Span width {span_width} is outside the range [{self.min_span_width}, {self.max_span_width}]")

                        projected_rep = self.projection(span_rep.squeeze(0))
                        
                        span_representations[i, j] = projected_rep

        return span_representations
