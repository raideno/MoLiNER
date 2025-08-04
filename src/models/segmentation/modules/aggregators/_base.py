import torch
import typing

from abc import ABC, abstractmethod

class BaseAggregator(torch.nn.Module, ABC):
    """
    Abstract base class for all aggregator modules.

    The role of an aggregator is to aggregate window-level predictions
    into frame-level predictions using various voting or scoring strategies.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        motion_length: int,
        window_metadata: torch.Tensor,
        class_probs: torch.Tensor,
        threshold: float,
        batch_index: typing.Optional[int] = None
    ) -> torch.Tensor:
        """
        Aggregate window-level predictions into frame-level predictions.

        Args:
            motion_length (int): Length of the motion sequence
            window_metadata (torch.Tensor): Window positions [num_windows, 3] -> [batch_idx, start_frame, end_frame]
            class_probs (torch.Tensor): Class probabilities [num_windows, num_classes]
            threshold (float): Confidence threshold for predictions
            batch_index (typing.Optional[int]): Optional batch index for logging/debugging

        Returns:
            torch.Tensor: Frame-level predictions with shape [motion_length]
                         Class indices with -1 for no prediction above threshold
        """
        pass
