import torch
import typing
import logging
import transformers

from .index import BasePromptsTokensEncoder

logger = logging.getLogger(__name__)

class TMRPromptsTokensEncoder(BasePromptsTokensEncoder):
    def __init__(
        self,
        frozen: bool = False,
        pretrained: bool = False,
        weights_path: typing.Optional[str] = None,
    ):
        super().__init__()
        
        self.frozen = frozen
        self.pretrained = pretrained
        self.weights_path = weights_path
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # NOTE: for feature extraction before passing it to the ActorStyle Encoder
        self.base_text_model = transformers.AutoModel.from_pretrained("distilbert-base-uncased")
        
        from src.model.helpers import ACTORStyleEncoder
        
        self.vae: bool = True
        self.latent_dim: int = 256
        self.ff_size: int = 1024
        self.num_layers: int = 6
        self.num_heads: int = 4
        self.dropout: float = 0.1
        self.activation: str = "gelu"
        
        self.tmr_text_encoder = ACTORStyleEncoder(
            # NOTE: DistilBERT hidden size
            nfeats=768,
            vae=True,
            latent_dim=self.latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation
        )
        
        if pretrained:
            if self.weights_path is not None:
                self.tmr_encoder.load_state_dict(
                    torch.load(weights_path)
                )
            else:
                logger.warning("Pretrained weights path is not provided. Using uninitialized TMR encoder.")
                raise ValueError("Pretrained weights path must be specified if pretrained is True.")
    
        if frozen:
            for param in self.tmr_text_encoder.parameters():
                param.requires_grad = False
        
        # NOTE: the base text model is always frozen
        for param in self.base_text_model.parameters():
            param.requires_grad = False
        
    def get_output_dim(self) -> int:
        """
        Returns the output dimension of the text encoder.
        Note: This might differ from latent_dim if the TMR encoder uses a different output dimension.
        """
        # NOTE: The TMR encoder might use a different output dimension than the configured latent_dim
        # We return latent_dim as a fallback, but the actual dimension is determined at runtime
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
        Returns the CLS token for each prompt.
        
        Args:
            prompt_input_ids (torch.Tensor): Token IDs of shape (batch_size, num_prompts, seq_len)
            prompt_attention_mask (torch.Tensor): Attention mask of shape (batch_size, num_prompts, seq_len)
            
        Returns:
            torch.Tensor: CLS token embeddings of shape (batch_size, num_prompts, hidden_size)
        """
        # NOTE: (batch_size, num_prompts, seq_len)
        B, P, L = prompt_input_ids.shape
        
        # NOTE: we reshape to (B * P, L) to process all prompts in one go
        reshaped_ids = prompt_input_ids.view(B * P, L)
        reshaped_mask = prompt_attention_mask.view(B * P, L)
        
        with torch.set_grad_enabled(not self.frozen):
            base_outputs = self.base_text_model(
                input_ids=reshaped_ids,
                attention_mask=reshaped_mask,
                return_dict=True
            )
            
            # NOTE: (B * P, L, 768)
            base_features = base_outputs.last_hidden_state
            
            cls_embeddings_list = []
            
            for i in range(B * P):
                # NOTE: we skip empty prompts
                if reshaped_mask[i].sum() == 0:
                    # NOTE: We'll determine the correct size after processing at least one valid prompt
                    cls_embeddings_list.append(None)
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
                    
                    # NOTE: cls_token is of shape (1, hidden_size) as tmr is a VAE, thus we only take the mean
                    mean, std = cls_token.squeeze(0)
                    cls_token = mean
                    
                    # NOTE: Use the CLS token directly (shape: [1, hidden_size])
                    cls_embedding = cls_token.squeeze(0)  # Remove batch dimension
                    cls_embeddings_list.append(cls_embedding)
                    
                except Exception as e:
                    logger.warning(f"Failed to encode prompt {i}, using zero CLS embedding: {e}")
                    # NOTE: We'll determine the correct size after processing at least one valid prompt
                    cls_embeddings_list.append(None)
            
            # NOTE: Find the first valid embedding to determine the correct dimension
            valid_embedding = None
            for embedding in cls_embeddings_list:
                if embedding is not None:
                    valid_embedding = embedding
                    break
            
            if valid_embedding is None:
                # NOTE: All prompts are empty, use the configured latent_dim
                actual_hidden_size = self.get_output_dim()
            else:
                actual_hidden_size = valid_embedding.shape[-1]
            
            # NOTE: Replace None entries with zero embeddings of the correct size
            for i, emb in enumerate(cls_embeddings_list):
                if emb is None:
                    cls_embeddings_list[i] = torch.zeros(
                        actual_hidden_size,
                        device=prompt_input_ids.device,
                        dtype=torch.float32
                    )
            
            # NOTE: stack all CLS embeddings: (B * P, hidden_size)
            flat_cls_embeddings = torch.stack(cls_embeddings_list, dim=0)
            
            # NOTE: Get the actual hidden size from the tensor
            actual_hidden_size = flat_cls_embeddings.shape[-1]
            
            # NOTE: Reshape back to (B, P, actual_hidden_size)
            output_embeddings = flat_cls_embeddings.view(B, P, actual_hidden_size)
            
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
        Tokenizes a list of text strings.
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
        return self.tokenizer.model_max_length
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id