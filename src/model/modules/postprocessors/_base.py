import torch
import typing

from abc import ABC, abstractmethod

from src.types import (
    ForwardOutput,
    EvaluationResult
)

class BasePostprocessor(torch.nn.Module, ABC):
    """
    Abstract base class for all postprocessor modules.

    The role of a postprocessor is to take in the decoded output from a decoder module
    and postprocess the spans to make some adjustments to the predictions before they undergo an evaluation.
    """
    
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        evaluation_results: EvaluationResult,
    ) -> EvaluationResult:
        """
        Postprocess the decoder's output.

        Args:
            evaluation_results (typing.List[EvaluationResult]).

        Returns:
            typing.List[EvaluationResult].
        """
        pass