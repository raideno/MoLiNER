import torch
import typing

from ._base import BaseSpanRepresentationLayer

class MLPSpanRepresentationLayer(BaseSpanRepresentationLayer):
    """
    A span representation layer that uses MLPs to aggregate frame embeddings.
    We define one MLP for each possible span width, or a single shared MLP.
    Each MLP takes the concatenated frame embeddings of a span and outputs the final representation.
    """

    def __init__(
        self,
        motion_embed_dim: int,
        representation_dim: int,
        max_span_width: int,
        min_span_width: int = 1,
        shared_weights: bool = False,
        hidden_dim: typing.Optional[int] = None,
        dropout_rate: float = 0.1,
        num_layers: int = 2
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
        self.motion_embed_dim = motion_embed_dim
        self.representation_dim = representation_dim
        
        if hidden_dim is None:
            hidden_dim = representation_dim

        if shared_weights:
            # NOTE: Shared MLP that works with the maximum possible input size (max_span_width * motion_embed_dim)
            # We pad smaller spans with zeros
            input_dim = max_span_width * motion_embed_dim
            self.mlp = self._create_mlp(input_dim, representation_dim, hidden_dim, dropout_rate, num_layers)
        else:
            # NOTE: Create one MLP for each possible span width
            self.mlps = torch.nn.ModuleList([
                self._create_mlp(
                    # input size = span_width * embed_dim
                    k * motion_embed_dim,
                    representation_dim,
                    hidden_dim,
                    dropout_rate,
                    num_layers
                ) for k in range(min_span_width, max_span_width + 1)
            ])

    def _create_mlp(self, input_dim: int, output_dim: int, hidden_dim: int, dropout_rate: float, num_layers: int) -> torch.nn.Module:
        if num_layers == 1:
            return torch.nn.Linear(input_dim, output_dim)
        
        layers = []
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout_rate))
        
        for _ in range(num_layers - 2):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
        
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        
        return torch.nn.Sequential(*layers)

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
        device = motion_features.device

        span_representations = torch.zeros(batch_size, max_spans, self.representation_dim, device=device)

        for i in range(batch_size):
            for j in range(max_spans):
                if spans_masks[i, j]:
                    start, end = span_indices[i, j]
                    # NOTE: end is inclusive
                    span_width = end - start + 1

                    if span_width > 0:
                        if self.min_span_width <= span_width <= self.max_span_width:
                            # (span_width, embed_dim)
                            span_frames = motion_features[i, start:end + 1]
                            # (span_width * embed_dim,)
                            span_flattened = span_frames.flatten()

                            if self.shared_weights:
                                # NOTE: pad to max_span_width with zeroes if necessary
                                if span_width < self.max_span_width:
                                    padding_size = (self.max_span_width - span_width) * self.motion_embed_dim
                                    padding = torch.zeros(padding_size, device=device)
                                    span_flattened = torch.cat([span_flattened, padding])
                                
                                span_rep = self.mlp(span_flattened)
                            else:
                                mlp_index = span_width - self.min_span_width
                                span_rep = self.mlps[mlp_index](span_flattened)
                        else:
                            raise ValueError(f"Span width {span_width} is outside the range [{self.min_span_width}, {self.max_span_width}]")

                        span_representations[i, j] = span_rep

        return span_representations