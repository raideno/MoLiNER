import torch
import typing
import transformers

from ._base import BasePromptsTokensEncoder

class CLIPPromptsTokensEncoder(BasePromptsTokensEncoder):
    """
    A prompts token encoder that uses the CLIP text encoder model.
    
    This encoder leverages CLIP's pretrained text encoder to produce
    contextual embeddings for text prompts, which are particularly
    well-suited for multimodal tasks involving text and motion.
    """
    
    def __init__(
        self,
        frozen: bool = True
    ):
        """
        Initializes the CLIPPromptsTokensEncoder.

        Args:
            frozen (bool): If True, the CLIP model's weights will not be updated during training.
        """
        super().__init__()
        
        MODEL_NAME = "openai/clip-vit-base-patch32"
        
        self.frozen = frozen
        
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(MODEL_NAME)
        self.text_encoder = transformers.CLIPTextModel.from_pretrained(MODEL_NAME)
        
        if self.frozen:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def get_output_dim(self) -> int:
        """
        Returns the hidden dimension size of the CLIP text encoder.
        """
        return self.text_encoder.config.hidden_size

    def forward(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        batch_index: typing.Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encodes prompts using the CLIP text encoder.
        
        Args:
            prompt_input_ids (torch.Tensor): Token IDs of shape (batch_size, num_prompts, seq_len)
            prompt_attention_mask (torch.Tensor): Attention mask of shape (batch_size, num_prompts, seq_len)
            
        Returns:
            torch.Tensor: Text embeddings of shape (batch_size, num_prompts, hidden_size)
        """
        # NOTE: (batch_size, num_prompts, seq_len)
        B, P, L = prompt_input_ids.shape
        
        # NOTE: reshape to (B * P, L) to process all prompts in one go
        reshaped_ids = prompt_input_ids.view(B * P, L)
        reshaped_mask = prompt_attention_mask.view(B * P, L)
        
        with torch.set_grad_enabled(not self.frozen):
            # CLIP text encoder returns pooled output (CLS-like representation)
            outputs = self.text_encoder(
                input_ids=reshaped_ids,
                attention_mask=reshaped_mask,
                return_dict=True
            )
            
            # NOTE: Use pooler_output which is the final representation after pooling
            # Shape: (batch_size * num_prompts, hidden_size)
            text_embeddings = outputs.pooler_output
        
        # NOTE: reshape back to (B, P, hidden_size)
        hidden_size = text_embeddings.shape[-1]
        output_embeddings = text_embeddings.view(B, P, hidden_size)
        
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
        """
        Tokenizes a list of text strings using CLIP tokenizer.
        
        Args:
            texts: List of text strings to tokenize
            max_length: Maximum sequence length (uses model default if None)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Format of returned tensors
            batch_index: Optional batch index (unused)
            
        Returns:
            Dictionary containing tokenized inputs
        """
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
        Returns the maximum sequence length supported by the CLIP tokenizer.
        """
        return self.tokenizer.model_max_length
    
    @property
    def pad_token_id(self) -> int:
        """
        Returns the padding token ID used by the CLIP tokenizer.
        """
        return self.tokenizer.pad_token_id
