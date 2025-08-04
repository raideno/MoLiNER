import torch
import typing
import logging

from ._base import BaseMotionEncoder

logger = logging.getLogger(__name__)

HML3D_FEATURES_SIZE = 263

class TMRMotionEncoder(BaseMotionEncoder):
    def __init__(
        self,
        frozen: bool = False,
        pretrained: bool = False,
        weights_path: typing.Optional[str] = None,
        sample_mean: bool = False
        # --- --- ---
    ):
        super().__init__()
        
        self.frozen = frozen
        self.pretrained_ = pretrained
        self.weights_path = weights_path
        
        from src.model.helpers import ACTORStyleEncoder
        
        self.vae: bool = True
        self.latent_dim: int = 256
        self.ff_size: int = 1024
        self.num_layers: int = 6
        self.num_heads: int = 4
        self.dropout: float = 0.1
        self.activation: str = "gelu"
        
        self.tmr_encoder = ACTORStyleEncoder(
            nfeats=HML3D_FEATURES_SIZE,
            vae=self.vae,
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
            for param in self.tmr_encoder.parameters():
                param.requires_grad = False
    
    def forward(
        self, 
        motion_features: torch.Tensor, 
        motion_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode motion features into frame-level embeddings.
        
        Args:
            motion_features (torch.Tensor): Input motion features of shape (batch_size, seq_len, 263)
            motion_masks (torch.Tensor): Mask indicating valid frames (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Encoded motion features of shape (batch_size, seq_len, latent_dim)
        """
        batch_size, seq_len, feat_dim = motion_features.shape
        
        inputs = {
            "x": motion_features.float(),
            "mask": motion_masks.bool()
        }
        
        cls_token, full_sequence = self.tmr_encoder.forward(
            x_dict=inputs,
            return_full_sequence=True
        )
        
        cls_tokens = full_sequence[:, :self.tmr_encoder.nbtokens, :]
        frame_embeddings = full_sequence[:, self.tmr_encoder.nbtokens:, :]
        
        mean, logvar = cls_tokens[:, 0, :], cls_tokens[:, 1, :]
        
        embedding = mean
        
        return embedding
    
    @property
    def pretrained(self) -> bool:
        """
        Indicates whether the encoder is pretrained or not.
        This is used by the model to adjust learning rates and training strategies.
        """
        return self.pretrained_
