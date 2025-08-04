import torch
import typing

from abc import ABC, abstractmethod

class BasePromptRepresentationLayer(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        aggregated_prompts: torch.Tensor,
        prompts_mask: torch.Tensor,
        batch_index: typing.Optional[int] = None
    ) -> torch.Tensor:
        """
        Transforms aggregated prompt vectors into their final representation.
        Args:
            aggregated_prompts (torch.Tensor): Aggregated vectors for each prompt.
            prompts_mask (torch.Tensor): Mask for padded prompts.
        Returns:
            torch.Tensor: The final prompt representations.
        """
        pass