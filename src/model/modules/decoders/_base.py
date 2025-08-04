import torch
import typing

from abc import ABC, abstractmethod

from src.types import (
    MolinerForwardOutput,
    EvaluationResult,
    RawBatch,
)

class BaseDecoder(torch.nn.Module, ABC):
    """
    Abstract base class for all decoder modules.

    The role of a decoder is to take the raw output from the model's forward pass
    and decode it into a list of predicted spans with their associated prompts.
    """
    
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        forward_output: MolinerForwardOutput,
        batch: RawBatch,
        score_threshold: float,
    ) -> EvaluationResult:
        """
        Decodes the model's forward pass output into a list of predicted spans.

        Args:
            forward_output: MolinerForwardOutput
            batch: RawBatch
            score_threshold: float,

        Returns:
            typing.List[EvaluationResult]: A list of EvaluationResult objects, one for each item in the batch.
        """
        pass