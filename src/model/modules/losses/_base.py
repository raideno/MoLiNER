import abc
import torch
import typing

from src.types import ForwardOutput, ProcessedBatch

class BaseLoss(torch.nn.Module, abc.ABC):
    """
    Abstract base class for all loss modules.

    The role of a loss function is to compute the complete training loss pipeline including:
    - Target matrix creation
    - Loss mask computation
    - Negatives masking
    - Final loss computation and reduction
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        forward_output: ForwardOutput,
        batch: ProcessedBatch,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        """
        Computes the complete loss given model forward output and processed batch.

        Args:
            forward_output (ForwardOutput): Model predictions and intermediate outputs
            batch (ProcessedBatch): Processed batch containing ground truth data
            batch_index (typing.Optional[int]): Optional index of the batch in the dataset, if applicable.

        Returns:
            loss (torch.Tensor): Computed loss value
        """
        pass