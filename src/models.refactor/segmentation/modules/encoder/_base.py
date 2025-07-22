import torch
import typing

from abc import ABC, abstractmethod

class BaseMotionEncoder(torch.nn.Module, ABC):
    """
    Abstract base class for all motion encoder modules.

    The role of an encoder is to process the original motion sequence and
    produce a fixed-size representation for the whole motion frame, which will be used
    in subsequent processing steps.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        motion_features: torch.Tensor,
        motion_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None
    ) -> torch.Tensor:
        """
        Encodes the motion into a fixed-size representation.

        Args:
            motion_features (torch.Tensor): The original motion frame embeddings.
                Shape: (batch_size, seq_len, embed_dim)
            motion_masks (torch.Tensor): A tensor indicating valid frames (1 or True) 
                vs. padding (0 or False).
                Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Encoded motion features.
                Shape: (batch_size, seq_len, encoded_dim)
        """
        pass
    
    @property
    @abstractmethod
    def pretrained(self) -> bool:
        """
        Indicates whether the encoder is pretrained or not.
        This is used by the model to adjust learning rates and training strategies.
        """
        pass