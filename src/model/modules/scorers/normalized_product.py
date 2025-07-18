import torch
import typing

from ._base import BasePairScorer

class NormalizedProductPairScorer(BasePairScorer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        # NOTE: (batch_size, num_prompts, representation_dim)
        prompts_representation: torch.Tensor,
        # NOTE: (batch_size, num_spans, representation_dim)
        spans_representation: torch.Tensor,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        if prompts_representation.shape[-1] != spans_representation.shape[-1]:
            raise ValueError(
                "Representation dimension for prompts and spans must be the same. "
                f"Got {prompts_representation.shape[-1]} and {spans_representation.shape[-1]}."
            )

        prompts_normalized = torch.nn.functional.normalize(prompts_representation, dim=-1)
        spans_normalized = torch.nn.functional.normalize(spans_representation, dim=-1)

        similarity_matrix = torch.einsum('bpd,bsd->bps', prompts_normalized, spans_normalized)
        
        return similarity_matrix
