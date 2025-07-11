import torch
import typing

from ._base import BaseSpanRepresentationLayer

class LSTMSpanRepresentationLayer(BaseSpanRepresentationLayer):
    """
    NOTE: A span representation layer that uses an LSTM to encode the frames within each span.
    The final hidden state of the LSTM is used as the span representation.
    """

    def __init__(self, motion_embed_dim: int, representation_dim: int, num_layers: int = 1, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=motion_embed_dim,
            # NOTE: we divide by 2 as for the bidirectional LSTM, the final state is a concatenation of the forward and backward states
            hidden_size=representation_dim // 2 if bidirectional else representation_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(
        self,
        motion_features: torch.Tensor,
        span_indices: torch.Tensor,
        spans_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, max_spans, _ = span_indices.shape
        representation_dim = self.lstm.hidden_size * (2 if self.lstm.bidirectional else 1)
        device = motion_features.device

        span_representations = torch.zeros(batch_size, max_spans, representation_dim, device=device)

        for i in range(batch_size):
            for j in range(max_spans):
                if spans_masks[i, j]:
                    start, end = span_indices[i, j]
                    # NOTE: end is inclusive, so we need to add 1
                    span_frames = motion_features[i:i+1, start:end + 1] # (1, span_width, embed_dim)

                    if span_frames.shape[1] > 0:
                        _, (hidden, _) = self.lstm(span_frames)
                        # NOTE: hidden is (num_layers * num_directions, batch, hidden_size)
                        # NOTE: we concatenate the final hidden states of the forward and backward LSTMs.
                        if self.lstm.bidirectional:
                            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
                        else:
                            hidden = hidden[-1, :, :]
                        span_representations[i, j] = hidden.squeeze(0)

        return span_representations
