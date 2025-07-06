import torch
import typing

from abc import ABC, abstractmethod

class BasePromptTokensAggregator(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        prompts_tokens_embeddings: torch.Tensor,
        prompts_attention_mask: torch.Tensor,
        batch_index: typing.Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Aggregates token embeddings into a single vector per prompt.
        Args:
            prompts_tokens_embeddings (torch.Tensor): Embeddings of all tokens.
            prompts_mask (torch.Tensor): Mask for padded prompts.
        Returns:
            torch.Tensor: A tensor of aggregated prompt representations.
        """
        pass