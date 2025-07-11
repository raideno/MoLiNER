import torch
import typing
import transformers

from ._base import BasePromptsTokensEncoder

class DebertaPromptsTokensEncoder(BasePromptsTokensEncoder):
    """
    A prompts token encoder that uses a pretrained transformer model from
    the Hugging Face Hub (e.g., DeBERTa, BERT).
    """
    def __init__(
        self,
        frozen: bool
    ):
        """
        Initializes the DebertaPromptsTokensEncoder.

        Args:
            frozen (bool): If True, the transformer's weights will not be updated during training.
        """
        super().__init__()
        
        MODEL_NAME = "microsoft/deberta-v3-base"
        
        self.frozen = frozen
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
        self.transformer = transformers.AutoModel.from_pretrained(MODEL_NAME)

        self.transformer.train()

        if self.frozen:
            self.transformer.eval()
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
        batch_index: typing.Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Passes the prompts through the transformer model to get CLS token embeddings.

        It handles the necessary reshaping to feed the 3D input
        (batch, num_prompts, seq_len) into the 2D-expecting transformer model.
        Returns the CLS token (first token) for each prompt.
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
        
        # NOTE: Extract CLS token (first token) for each prompt
        # Shape: (batch_size * num_prompts, hidden_size)
        cls_tokens = last_hidden_state[:, 0, :]
        
        # NOTE: reshape back to (B, P, hidden_size)
        hidden_size = cls_tokens.shape[-1]
        output_embeddings = cls_tokens.view(B, P, hidden_size)
        
        return output_embeddings
    
    def tokenize(
        self, 
        texts: typing.List[str],
        max_length: typing.Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        batch_index: typing.Optional[int] = None,
    ) -> typing.Dict[str, torch.Tensor]:
        if max_length is None:
            max_length = self.model_max_length
            
        result = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        
        return result
    
    @property
    def model_max_length(self) -> int:
        """
        Returns the maximum sequence length supported by the Deberta tokenizer.
        """
        # return self.tokenizer.model_max_length
        return 512
    
    @property
    def pad_token_id(self) -> int:
        """
        Returns the padding token ID used by the Deberta tokenizer.
        """
        return self.tokenizer.pad_token_id