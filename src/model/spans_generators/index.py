import torch
import typing

from abc import ABC, abstractmethod

class BaseSpansGenerator(torch.nn.Module, ABC):
    """
    Abstract base class for all span generator modules.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        motion_features: torch.Tensor,
        motion_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes motion features to generate candidate spans.
        This simplified version returns padded tensors for easier batch processing.

        Args:
            motion_features (torch.Tensor): A tensor of motion frame embeddings.
                Shape: (batch_size, seq_len, embed_dim)
            motion_masks (torch.Tensor): A tensor indicating valid frames (1 or True) 
                vs. padding (0 or False).
                Shape: (batch_size, seq_len)

        Returns:
            A tuple containing two tensors:
            
            - span_indices (torch.Tensor): A tensor of [start, end] frame indices for all
              generated spans in the batch. Padded with -1.
              Shape: (batch_size, max_num_spans, 2)
              
            - span_mask (torch.Tensor): A boolean tensor indicating which spans are valid (True)
              vs. padding (False).
              Shape: (batch_size, max_num_spans)
        """
        pass