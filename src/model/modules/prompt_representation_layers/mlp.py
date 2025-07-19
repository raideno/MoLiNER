import torch
import typing

from ._base import BasePromptRepresentationLayer

from src.model.modules._helpers import create_projection_layer

class MLPPromptRepresentationLayer(BasePromptRepresentationLayer):
    def __init__(self, input_dim: int, representation_dim: int, dropout: float):
        super().__init__()
        
        self.prompt_rep_layer = create_projection_layer(input_dim, dropout, representation_dim)

    def forward(
        self,
        aggregated_prompts: torch.Tensor,
        prompts_mask: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        representations = self.prompt_rep_layer(aggregated_prompts)
        
        # NOTE: (Batch Size, #Prompts)
        # A prompt is valid if it has at least one non-padding token
        prompt_level_mask = prompts_mask.any(dim=-1)
        
        masked_representations = representations * prompt_level_mask.unsqueeze(-1)
        
        return masked_representations