import torch
import typing
import logging

from .index import BaseMotionFramesEncoder

logger = logging.getLogger(__name__)

HML3D_FEATURES_SIZE = 263

class TMRMotionFramesEncoder(BaseMotionFramesEncoder):
    """
    TMR-based motion frames encoder that processes motion sequences using a transformer architecture.
    
    This encoder uses the TMR (Text-Motion Retrieval) model's transformer-based encoder to generate
    frame-level embeddings from motion sequences.
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        pretrained: bool = False,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize the TMR motion frames encoder.
        
        Args:
            latent_dim (int): Dimension of the output embeddings
            pretrained (bool): Whether to use pretrained weights
            ff_size (int): Feed-forward network size in transformer layers
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
            activation (str): Activation function type
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        from src.model.helpers import ACTORStyleEncoder
        
        self.tmr_encoder = ACTORStyleEncoder(
            nfeats=HML3D_FEATURES_SIZE,
            vae=False,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation
        )
        
        if pretrained:
            # TODO: load the pretrained weights
            self.tmr_encoder.eval()
            raise NotImplementedError(
                "Pretrained weights loading is not implemented yet for TMRMotionFramesEncoder."
            )
    
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
        
        # NOTE: skip the cls token & extract frame embeddings only
        num_cls_tokens = self.tmr_encoder.nbtokens
        frame_embeddings = full_sequence[:, num_cls_tokens:, :]
        
        frame_embeddings = frame_embeddings * motion_masks.unsqueeze(-1).float()
        
        return frame_embeddings