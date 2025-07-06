import torch

from transformers import AutoModel

from .index import BasePromptsTokensEncoder

class TransformerPromptsTokensEncoder(BasePromptsTokensEncoder):
    """
    A prompts token encoder that uses a pretrained transformer model from
    the Hugging Face Hub (e.g., DeBERTa, BERT).
    """
    def __init__(self, model_name: str, finetune: bool = True):
        """
        Initializes the TransformerPromptsTokensEncoder.

        Args:
            model_name (str): The name of the model to load from the HuggingFace Hub (e.g., "microsoft/deberta-v3-base").
            finetune (bool): If True, the transformer's weights will be updated during training. If False, they will be frozen.
        """
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

        if not finetune:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def get_output_dim(self) -> int:
        """
        Returns the hidden dimension size of the transformer model.
        """
        return self.transformer.config.hidden_size

    def forward(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Passes the prompts through the transformer model to get contextual embeddings.

        It handles the necessary reshaping to feed the 3D input
        (batch, num_prompts, seq_len) into the 2D-expecting transformer model.
        """
        # NOTE: (batch_size, num_prompts, seq_len)
        B, P, L = prompt_input_ids.shape
        
        # NOTE: reshape to (B * P, L) to process all prompts in one go.
        reshaped_ids = prompt_input_ids.view(B * P, L)
        reshaped_mask = prompt_attention_mask.view(B * P, L)
        
        outputs = self.transformer(
            input_ids=reshaped_ids,
            attention_mask=reshaped_mask,
            return_dict=True
        )
        
        # NOTE: (batch_size * num_prompts, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        
        # NOTE: reshape back to (B, P, L, hidden_size)
        hidden_size = last_hidden_state.shape[-1]
        output_embeddings = last_hidden_state.view(B, P, L, hidden_size)
        
        return output_embeddings