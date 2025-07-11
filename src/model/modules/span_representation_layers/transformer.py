import torch
import typing

from ._base import BaseSpanRepresentationLayer

class TransformerSpanRepresentationLayer(BaseSpanRepresentationLayer):
    """
    A span representation layer that uses a Transformer encoder to process frames within a span.
    A [CLS] token is prepended to the span's frames, and its corresponding output embedding is used as the span representation.
    """

    def __init__(
        self,
        motion_embed_dim: int,
        representation_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, motion_embed_dim))
        
        # TODO: should we add a projection layer first ?
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=motion_embed_dim,
            nhead=num_heads,
            dim_feedforward=representation_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.output_projection = torch.nn.Linear(motion_embed_dim, representation_dim)

    def forward(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        spans_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, max_spans, _ = span_indices.shape
        representation_dim = self.output_projection.out_features
        device = motion_features.device

        span_representations = torch.zeros(batch_size, max_spans, representation_dim, device=device)

        # NOTE: we can't perform batch processing as spans are of different lengths
        # TODO: what could be done is to group spans of similar lengths together and process them in batches, but for now we will process one span at a time as its simpler
        for i in range(batch_size):
            for j in range(max_spans):
                if spans_masks[i, j]:
                    start, end = span_indices[i, j]
                    # NOTE: end is inclusive, so we need to add 1
                    span_frames = motion_features[i:i+1, start:end + 1] # (1, span_width, embed_dim)

                    if span_frames.shape[1] > 0:
                        # NOTE: Prepend [CLS] token
                        cls_token = self.cls_token.expand(1, -1, -1)
                        span_with_cls = torch.cat([cls_token, span_frames], dim=1)
                        
                        encoded_span = self.transformer_encoder(span_with_cls)
                        
                        # NOTE: Use the embedding of the [CLS] token as the representation
                        cls_embedding = encoded_span[:, 0, :]
                        
                        span_representations[i, j] = self.output_projection(cls_embedding).squeeze(0)

        return span_representations
