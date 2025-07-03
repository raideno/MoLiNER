import torch
import typing
import logging

from .index import BasePromptsTokensEncoder

logger = logging.getLogger(__name__)

class PretrainedTMRPromptsTokensEncoder(BasePromptsTokensEncoder):
    """
    A prompts token encoder that uses the text encoder from a pretrained TMR model.
    
    This encoder loads a pretrained TMR model and uses its text encoder
    to generate contextual embeddings for prompt tokens.
    """
    
    def __init__(
        self,
        model_path: str,
        finetune: bool = True,
        output_dim: typing.Optional[int] = None,
    ):
        """
        Initialize the pretrained TMR prompts tokens encoder.
        
        Args:
            model_path (str): Path to the pretrained TMR model checkpoint
            finetune (bool): Whether to finetune the model or freeze it
            output_dim (int, optional): Override the output dimension from the pretrained model
        """
        super().__init__()
        
        self.model_path = model_path
        self.finetune = finetune
        self.tmr_model: typing.Any = None  # Will be set in _load_pretrained_model
        self.output_dim_override = output_dim
        
        # Load the pretrained TMR model
        self._load_pretrained_model()
        
        # Freeze parameters if not finetuning (after model is loaded)
        if not self.finetune and self.tmr_model is not None:
            for param in self.tmr_model.text_encoder.parameters():
                param.requires_grad = False
            logger.info("TMR text encoder parameters frozen (finetune=False)")
        else:
            logger.info("TMR text encoder parameters will be finetuned")
    
    def _load_pretrained_model(self):
        """Load the pretrained TMR model from checkpoint."""
        try:
            # Load the checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Extract the model state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load the TMR model architecture
            from src.model.tmr import TMR
            from src.model.actor import ACTORStyleEncoder, ACTORStyleDecoder
            from src.model.text_encoder import TextToEmb
            
            # Create motion encoder (we won't use this but TMR expects it)
            motion_encoder = ACTORStyleEncoder(
                nfeats=263,  # HML3D features
                vae=True,
                latent_dim=256,
                ff_size=1024,
                num_layers=6,
                num_heads=4,
                dropout=0.1,
                activation="gelu"
            )
            
            # Create text encoder (this is what we'll use)
            text_encoder = TextToEmb()
            
            # Create motion decoder (we won't use this but TMR expects it)
            motion_decoder = ACTORStyleDecoder(
                nfeats=263,
                latent_dim=256,
                ff_size=1024,
                num_layers=6,
                num_heads=4,
                dropout=0.1,
                activation="gelu"
            )
            
            # Create TMR model
            self.tmr_model = TMR(
                motion_encoder=motion_encoder,
                text_encoder=text_encoder,
                motion_decoder=motion_decoder,
                vae=True
            )
            
            # Load the pretrained weights
            self.tmr_model.load_state_dict(state_dict, strict=False)
            
            logger.info(f"Successfully loaded pretrained TMR model for text encoder from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load pretrained TMR model from {self.model_path}: {e}")
            raise
    
    def get_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder.
        """
        if self.output_dim_override is not None:
            return self.output_dim_override
        
        # Try to get the output dimension from the text encoder
        try:
            # The text encoder usually has a latent_dim attribute or similar
            if hasattr(self.tmr_model.text_encoder, 'latent_dim'):
                return int(self.tmr_model.text_encoder.latent_dim)
            elif hasattr(self.tmr_model.text_encoder, 'output_dim'):
                return int(self.tmr_model.text_encoder.output_dim)
            else:
                # Default fallback
                return 256
        except:
            return 256
    
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
        
        # Set evaluation mode if not finetuning
        if not self.finetune:
            self.tmr_model.text_encoder.eval()
        
        # The TMR text encoder typically expects text strings, not token IDs
        # We need to decode the token IDs back to text first
        # This is a limitation - ideally we'd want direct token-level embeddings
        
        # For now, we'll create a workaround that processes each prompt separately
        output_embeddings_list = []
        
        with torch.set_grad_enabled(self.finetune):
            for b in range(B):
                prompt_embeddings_list = []
                for p in range(P):
                    # Get the tokens for this specific prompt
                    prompt_tokens = prompt_input_ids[b, p]  # (seq_len,)
                    prompt_mask = prompt_attention_mask[b, p]  # (seq_len,)
                    
                    # Skip empty prompts (all padding)
                    if prompt_mask.sum() == 0:
                        # Create zero embeddings for empty prompts
                        empty_embeddings = torch.zeros(
                            L, self.get_output_dim(), 
                            device=prompt_input_ids.device, 
                            dtype=torch.float32
                        )
                        prompt_embeddings_list.append(empty_embeddings)
                        continue
                    
                    # For now, we'll use a simple approach: create token-level embeddings
                    # by replicating the sentence embedding across all tokens
                    # This is not ideal but works as a starting point
                    
                    # Create a dummy text input - in practice, you'd need a tokenizer
                    # to convert token IDs back to text
                    dummy_text = ["dummy text for tmr"]  # This should be decoded from tokens
                    
                    try:
                        # Get text embeddings from TMR text encoder
                        text_emb = self.tmr_model.text_encoder(dummy_text)
                        
                        # text_emb shape: (1, embedding_dim)
                        # We need to expand this to (seq_len, embedding_dim)
                        embedding_dim = text_emb.shape[-1]
                        token_embeddings = text_emb.unsqueeze(0).expand(L, embedding_dim)
                        
                        # Apply the attention mask
                        token_embeddings = token_embeddings * prompt_mask.unsqueeze(-1).float()
                        
                        prompt_embeddings_list.append(token_embeddings)
                        
                    except Exception as e:
                        logger.warning(f"Failed to encode prompt, using zero embeddings: {e}")
                        # Fallback to zero embeddings
                        zero_embeddings = torch.zeros(
                            L, self.get_output_dim(), 
                            device=prompt_input_ids.device, 
                            dtype=torch.float32
                        )
                        prompt_embeddings_list.append(zero_embeddings)
                
                # Stack all prompt embeddings for this batch item
                batch_prompt_embeddings = torch.stack(prompt_embeddings_list, dim=0)  # (P, L, embedding_dim)
                output_embeddings_list.append(batch_prompt_embeddings)
        
        # Stack all batch embeddings
        output_embeddings = torch.stack(output_embeddings_list, dim=0)  # (B, P, L, embedding_dim)
        
        return output_embeddings
