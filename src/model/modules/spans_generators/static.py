import torch
import typing

from ._base import BaseSpansGenerator

from src.constants import (
    DEFAULT_PADDING_VALUE
)

class StaticSpansGenerator(BaseSpansGenerator):
    def __init__(self, min_width: int, max_width: int, padding_value: int = int(DEFAULT_PADDING_VALUE)):
        """
        Initializes the StaticSpansGenerator.

        Args:
            max_width (int): The maximum length of a span to generate. Spans of all lengths from min_width to max_width will be generated.
            min_width (int): The minimum length of a span to generate.
            padding_value (int): The value to use for padding the span indices tensor. Defaults to src/constants.py:DEFAULT_PADDING_VALUE.
        """
        super().__init__()
        
        if not isinstance(max_width, int) or max_width < 1:
            raise ValueError("max_width must be a positive integer.")
        if not isinstance(min_width, int) or min_width < 1:
            raise ValueError("min_width must be a positive integer.")
        if min_width > max_width:
            raise ValueError("min_width must be less than or equal to max_width.")
        
        self.max_width = max_width
        self.min_width = min_width
        self.padding_value = padding_value
        
    def forward(
        self,
        # (batch_size, seq_len, embed_dim)
        motion_features: torch.Tensor,
        # (batch_size, seq_len)
        motion_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates all possible contiguous spans from min_width to max_width and returns them as padded tensors.

        Returns:
            A tuple containing two tensors:
            
            - span_indices (torch.Tensor): A tensor of [start, end] frame indices for all
              generated spans in the batch. Padded with `self.padding_value`.
              Shape: (batch_size, max_num_spans, 2)
              
            - span_mask (torch.Tensor): A boolean tensor indicating which spans are valid (True)
              vs. padding (False).
              Shape: (batch_size, max_num_spans)
        """
        batch_size, _, _ = motion_features.shape
        device = motion_features.device

        batch_spans_positions = []
        span_lengths = []

        for i in range(batch_size):
            current_length = int(motion_masks[i].sum().item())
            sample_spans_positions = []
            
            if current_length > 0:
                for start_idx in range(current_length):
                    for length in range(self.min_width, self.max_width + 1):
                        end_idx = start_idx + length - 1
                        if end_idx >= current_length:
                            break
                        sample_spans_positions.append([start_idx, end_idx])

            span_lengths.append(len(sample_spans_positions))
            
            if sample_spans_positions:
                batch_spans_positions.append(torch.tensor(sample_spans_positions, device=device, dtype=torch.long))
            else:
                batch_spans_positions.append(torch.empty(0, 2, device=device, dtype=torch.long))

        padded_span_indices = torch.nn.utils.rnn.pad_sequence(
            batch_spans_positions,
            batch_first=True,
            padding_value=self.padding_value
        )
        
        max_spans = padded_span_indices.shape[1]
        
        span_mask = torch.arange(max_spans, device=device)[None, :] < torch.tensor(span_lengths, device=device)[:, None]

        return padded_span_indices, span_mask