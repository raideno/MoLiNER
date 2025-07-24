import torch
import typing

from abc import ABC, abstractmethod

from src.types import (
    ForwardOutput,
    DecodingStrategy,
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
    def decode(
        self,
        evaluation_results: typing.List[EvaluationResult],
    ) -> typing.List[EvaluationResult]:
        """
        Postprocess the decoder's output.

        Args:
            evaluation_results (typing.List[EvaluationResult]): A list of EvaluationResult objects, one for each item in the batch.

        Returns:
            typing.List[EvaluationResult]: A list of EvaluationResult objects, one for each item in the batch.
        """
        pass