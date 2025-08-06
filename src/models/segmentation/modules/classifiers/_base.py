import torch
import typing

from abc import ABC, abstractmethod

class BaseClassifier(torch.nn.Module, ABC):
    """
    Abstract base class for all classifier modules.

    The role of a classifier is to process the encoded motion features and
    produce classification logits and boundary regression outputs for segmentation.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        encoded_features: torch.Tensor,
        motion_masks: typing.Optional[torch.Tensor] = None,
        batch_index: typing.Optional[int] = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Classifies the encoded motion features and predicts segment boundaries.

        Args:
            encoded_features (torch.Tensor): The encoded motion features from the encoder.
                Shape: (batch_size, latent_dim)
            motion_masks (torch.Tensor, optional): A tensor indicating valid frames (1 or True) 
                vs. padding (0 or False). Shape: (batch_size, seq_len)
            batch_index (int, optional): Index of the current batch, useful for debugging.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - class_logits: Classification logits. Shape: (batch_size, 1)
                - start_logits: Start boundary regression outputs. Shape: (batch_size, 1)
                - end_logits: End boundary regression outputs. Shape: (batch_size, 1)
        """
        pass