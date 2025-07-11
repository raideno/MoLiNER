import torch
import typing

from ._base import BasePairScorer

class ProductPairScorer(BasePairScorer):
    """
    A pair scorer that computes similarity as a the dot product.
    """
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
        """
        Performs a batched matrix multiplication between prompts and transposed spans,
        then applies a sigmoid function to produce similarity scores between 0 and 1.
        """
        if prompts_representation.shape[-1] != spans_representation.shape[-1]:
            raise ValueError(
                "Representation dimension for prompts and spans must be the same. "
                f"Got {prompts_representation.shape[-1]} and {spans_representation.shape[-1]}."
            )

        similarity_matrix = torch.einsum('bpd,bsd->bps', prompts_representation, spans_representation)

        return similarity_matrix