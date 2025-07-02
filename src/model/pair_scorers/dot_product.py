import torch

from .index import BasePairScorer

class DotProductPairScorer(BasePairScorer):
    """
    A pair scorer that computes similarity as a sigmoid of the dot product
    between prompt and span representations.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        # NOTE: (batch_size, num_prompts, representation_dim)
        prompts_representation: torch.Tensor,
        # NOTE: (batch_size, num_spans, representation_dim)
        spans_representation: torch.Tensor
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

        # # NTOE: (batch_size, representation_dim, num_spans)
        # spans_t = spans_representation.transpose(-2, -1)

        # # NOTE: Batched matrix multiplication (equivalent to dot product for each pair).
        # dot_product_scores = torch.matmul(prompts_representation, spans_t)

        # similarity_matrix = torch.sigmoid(dot_product_scores)

        similarity_matrix = torch.einsum('bpd,bsd->bps', prompts_representation, spans_representation)

        return similarity_matrix