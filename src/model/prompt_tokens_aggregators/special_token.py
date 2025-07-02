import torch

from .index import BasePromptTokensAggregator

class SpecialTokenAggregator(BasePromptTokensAggregator):
    """
    Aggregates prompts by selecting the embedding of the special token.
    """
    def forward(self, prompts_tokens_embeddings: torch.Tensor, prompts_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Aggregates token embeddings by selecting special token embeddings.
        
        Args:
            prompts_tokens_embeddings (torch.Tensor): Token embeddings of shape (batch_size, num_prompts, seq_len, embed_dim)
            prompts_attention_mask (torch.Tensor): Attention mask of shape (batch_size, num_prompts, seq_len) or
                                                   special token positions of shape (batch_size, num_prompts)
            **kwargs: Additional keyword arguments
            
        Returns:
            torch.Tensor: Aggregated prompt representations of shape (batch_size, num_prompts, embed_dim)
        """
        batch_size, num_prompts, seq_len, embed_dim = prompts_tokens_embeddings.shape
        
        special_token_positions = torch.zeros(batch_size, num_prompts, dtype=torch.long, device=prompts_tokens_embeddings.device)
        
        batch_indices = torch.arange(batch_size, device=prompts_tokens_embeddings.device).unsqueeze(1).expand(-1, num_prompts)
        prompt_indices = torch.arange(num_prompts, device=prompts_tokens_embeddings.device).unsqueeze(0).expand(batch_size, -1)
        
        # NOTE: (batch_size, num_prompts, embed_dim)
        aggregated_embeddings = prompts_tokens_embeddings[batch_indices, prompt_indices, special_token_positions]
        
        return aggregated_embeddings
