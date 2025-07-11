import torch
import typing

from abc import ABC, abstractmethod

class BasePromptsTokensEncoder(torch.nn.Module, ABC):
    """
    Abstract base class for all prompts tokens encoders.

    The role of an encoder is to take token IDs and produce contextual
    vector representations for each token.
    """
    @abstractmethod
    def forward(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        batch_index: typing.Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encodes prompt token IDs into dense, contextual embeddings.
        Returns one embedding per prompt.

        Args:
            prompt_input_ids (torch.Tensor): Padded tensor of token IDs.
                Shape: (batch_size, num_prompts, seq_len)
            prompt_attention_mask (torch.Tensor): The attention mask corresponding
                to the input IDs.
                Shape: (batch_size, num_prompts, seq_len)

        Returns:
            torch.Tensor: The resulting contextual embeddings (CLS token for each prompt).
                Shape: (batch_size, num_prompts, hidden_dim)
        """
        pass
    
    @abstractmethod
    def tokenize(
        self,
        texts: typing.List[str],
        max_length: typing.Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = 'pt'
    ) -> typing.Dict[str, torch.Tensor]:
        pass
    
    @property
    @abstractmethod
    def model_max_length(self) -> int:
        pass
    
    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        pass