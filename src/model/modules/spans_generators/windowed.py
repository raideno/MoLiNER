import torch
import typing

from ._base import BaseSpansGenerator

class WindowedSpansGenerator(BaseSpansGenerator):
    def __init__(self, K: int, stride: int = 1, padding_value: int = -1):
        """
        Initializes the WindowedSpansGenerator.
        
        This generator creates sliding windows of fixed size K across the motion sequence.
        Each window starts at a frame position and spans exactly K frames.
        
        Total Number of Spans = max(0, (N - K + stride) // stride) where N is sequence length
        
        Args:
            K (int): The fixed size of each window (number of frames per span).
            stride (int): The step size between consecutive windows. Defaults to 1.
            padding_value (int): The value to use for padding the span indices tensor. Defaults to -1.
        """
        super().__init__()
        
        if not isinstance(K, int) or K < 1:
            raise ValueError("K must be a positive integer.")
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a positive integer.")
        
        self.K = K
        self.stride = stride
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
        Generates sliding windows of fixed size K across the motion sequence.

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
            
            if current_length >= self.K:
                # NOTE: generate sliding windows of size K
                for start_idx in range(0, current_length - self.K + 1, self.stride):
                    end_idx = start_idx + self.K - 1  # end_idx is inclusive
                    
                    if end_idx < current_length:
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
