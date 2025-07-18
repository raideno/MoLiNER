import torch
import typing

from ._base import BasePromptRepresentationLayer

class MLPPromptRepresentationLayer(BasePromptRepresentationLayer):
    """
    A prompt representation layer that uses a two-layer MLP.
    """
    def __init__(self, input_dim: int, representation_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, representation_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(representation_dim * 4, representation_dim),
        )

    def forward(
        self,
        aggregated_prompts: torch.Tensor,
        prompts_mask: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        representations = self.mlp(aggregated_prompts)
        
        # NOTE: (Batch Size, #Prompts)
        # A prompt is valid if it has at least one non-padding token
        prompt_level_mask = prompts_mask.any(dim=-1)
        
        masked_representations = representations * prompt_level_mask.unsqueeze(-1)
        
        return masked_representations