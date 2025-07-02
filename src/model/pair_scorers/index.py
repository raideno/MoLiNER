import torch
import typing

from abc import ABC, abstractmethod

class BasePairScorer(torch.nn.Module, ABC):
    """
    Abstract base class for all pair scorer modules.

    The role of a scorer is to take the final representations of prompts and
    spans and compute a similarity score for each possible pair.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        prompts_representation: torch.Tensor,
        spans_representation: torch.Tensor,
        batch_index: typing.Optional[int] = None
    ) -> torch.Tensor:
        """
        Calculates a similarity matrix between prompts and spans.

        Args:
            prompts_representation (torch.Tensor): The final representations for all
                prompts in the batch.
                Shape: (batch_size, num_prompts, representation_dim)
            spans_representation (torch.Tensor): The final representations for all
                spans in the batch.
                Shape: (batch_size, num_spans, representation_dim)

        Returns:
            torch.Tensor: A matrix of similarity scores.
                Shape: (batch_size, num_prompts, num_spans)
        """
        pass