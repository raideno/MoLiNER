import torch
import typing
import logging

from transformers import AutoModel

from .index import BasePromptsTokensEncoder

logger = logging.getLogger(__name__)

class PretrainedTMRPromptsTokensEncoder(BasePromptsTokensEncoder):
    """
    A prompts token encoder that uses the text encoder from a pretrained TMR model.
    
    This encoder uses the TMR model's text encoder to generate contextual embeddings
    for prompt tokens. It expects pre-tokenized input (token IDs and attention masks)
    and focuses only on the encoding part, following the separation of concerns pattern.
    
    Note: Tokenization should be handled separately using a compatible tokenizer
    (e.g., DistilBertTokenizer) that matches the base model used here.
    """
    
    def __init__(
        self,
        weights_path: str,
        finetune: bool = True,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize the pretrained TMR prompts tokens encoder.
        
        Args:
            weights_path (str): Path to the pretrained TMR text encoder weights
            finetune (bool): Whether to finetune the model or freeze it
            latent_dim (int): Dimension of the output embeddings
            ff_size (int): Feed-forward network size in transformer layers
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
            activation (str): Activation function type
        """
        super().__init__()
        
        self.weights_path = weights_path
        self.finetune = finetune
        self.latent_dim = latent_dim
        
        # NOTE: for feature extraction before passing it to the ActorStyle Encoder
        self.base_text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        
        from src.model.helpers import ACTORStyleEncoder
        
        self.tmr_text_encoder = ACTORStyleEncoder(
            # NOTE: DistilBERT hidden size
            nfeats=768,
            vae=True,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation
        )
        
        self.tmr_text_encoder.load_state_dict(
            torch.load(weights_path, map_location='cpu')
        )
        self.tmr_text_encoder.train()
        
        if not finetune:
            for param in self.tmr_text_encoder.parameters():
                param.requires_grad = False
            for param in self.base_text_model.parameters():
                param.requires_grad = False
            logger.info("TMR text encoder parameters frozen (finetune=False)")
        else:
            logger.info("TMR text encoder parameters will be finetuned")
    
    def get_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder.
        """
        return self.latent_dim
    
    def forward(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        batch_index: typing.Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode prompts using the pretrained TMR text encoder.
        
        Args:
            prompt_input_ids (torch.Tensor): Token IDs of shape (batch_size, num_prompts, seq_len)
            prompt_attention_mask (torch.Tensor): Attention mask of shape (batch_size, num_prompts, seq_len)
            
        Returns:
            torch.Tensor: Encoded prompt tokens of shape (batch_size, num_prompts, seq_len, hidden_size)
        """
        # NOTE: (batch_size, num_prompts, seq_len)
        B, P, L = prompt_input_ids.shape
        
        # NOTE: we reshape to (B * P, L) to process all prompts in one go
        reshaped_ids = prompt_input_ids.view(B * P, L)
        reshaped_mask = prompt_attention_mask.view(B * P, L)
        
        with torch.set_grad_enabled(self.finetune):
            base_outputs = self.base_text_model(
                input_ids=reshaped_ids,
                attention_mask=reshaped_mask,
                return_dict=True
            )
            
            # NOTE: (B * P, L, 768)
            base_features = base_outputs.last_hidden_state
            
            output_embeddings_list = []
            
            for i in range(B * P):
                # NOTE: we skip empty prompts
                if reshaped_mask[i].sum() == 0:
                    empty_embeddings = torch.zeros(
                        L, self.get_output_dim(), 
                        device=prompt_input_ids.device, 
                        dtype=torch.float32
                    )
                    output_embeddings_list.append(empty_embeddings)
                    continue
                
                try:
                    prompt_features = base_features[i:i+1]
                    prompt_mask = reshaped_mask[i:i+1]
                    
                    text_input = {
                        "x": prompt_features,
                        "mask": prompt_mask.bool()
                    }
                    
                    cls_token, full_sequence = self.tmr_text_encoder.forward(
                        x_dict=text_input,
                        return_full_sequence=True
                    )
                    
                    num_cls_tokens = getattr(self.tmr_text_encoder, 'nbtokens', 0)
                    if num_cls_tokens > 0:
                        sequence_embeddings = full_sequence[:, num_cls_tokens:, :]
                    else:
                        sequence_embeddings = full_sequence
                    
                    # TODO: recheck the following padding logic
                    seq_len = sequence_embeddings.shape[1]
                    if seq_len < L:
                        padding = torch.zeros(
                            1, L - seq_len, self.get_output_dim(),
                            device=sequence_embeddings.device,
                            dtype=sequence_embeddings.dtype
                        )
                        sequence_embeddings = torch.cat([sequence_embeddings, padding], dim=1)
                    elif seq_len > L:
                        sequence_embeddings = sequence_embeddings[:, :L, :]
                    
                    token_embeddings = sequence_embeddings.squeeze(0)
                    token_embeddings = token_embeddings * reshaped_mask[i].unsqueeze(-1).float()
                    
                    output_embeddings_list.append(token_embeddings)
                    
                except Exception as e:
                    logger.warning(f"Failed to encode prompt {i}, using zero embeddings: {e}")
                    zero_embeddings = torch.zeros(
                        L, self.get_output_dim(), 
                        device=prompt_input_ids.device, 
                        dtype=torch.float32
                    )
                    output_embeddings_list.append(zero_embeddings)
            
            flat_embeddings = torch.stack(output_embeddings_list, dim=0)
            
            hidden_size = flat_embeddings.shape[-1]
            output_embeddings = flat_embeddings.view(B, P, L, hidden_size)
            
            return output_embeddings
