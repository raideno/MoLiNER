import abc
import torch
import typing

from src.types import SegmenterForwardOutput, RawBatch

class BaseLoss(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        forward_output: SegmenterForwardOutput,
        batch: RawBatch,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        """
        Computes the complete loss given model forward output and processed batch.

        Args:
            forward_output (SegmenterForwardOutput): Model predictions and intermediate outputs
            batch (RawBatch): Batch containing ground truth data
            batch_index (typing.Optional[int]): Optional index of the batch in the dataset, if applicable.

        Returns:
            loss (torch.Tensor): Computed loss value
        """
        pass
