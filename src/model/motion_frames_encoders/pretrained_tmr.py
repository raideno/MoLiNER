import torch
import typing
import logging

from src.model.motion_frames_encoders.index import BaseMotionFramesEncoder

logger = logging.getLogger(__name__)

HML3D_FEATURES_SIZE = 263

class PretrainedTMRMotionFramesEncoder(BaseMotionFramesEncoder):
    """
    TMR-based motion frames encoder that processes motion sequences using a transformer architecture.
    
    This encoder uses the TMR (Text-Motion Retrieval) model's transformer-based encoder to generate
    frame-level embeddings from motion sequences.
    """
    
    def __init__(
        self,
        weights_path: str,
        finetune: bool = True,
        latent_dim: int = 256,
        pretrained: bool = False,
        ff_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu",
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
        
        self.weights_path = weights_path
        self.finetune = finetune
        
        from src.model import ACTORStyleEncoder
        
        self.tmr_encoder = ACTORStyleEncoder(
            nfeats=HML3D_FEATURES_SIZE,
            vae=True,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation
        )
        self.tmr_encoder.load_state_dict(
            torch.load(weights_path)
        )
        self.tmr_encoder.train()
        
        if not finetune:
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
        
        # NOTE: skip the cls token & extract frame embeddings only
        num_cls_tokens = self.tmr_encoder.nbtokens
        frame_embeddings = full_sequence[:, num_cls_tokens:, :]
        
        frame_embeddings = frame_embeddings * motion_masks.unsqueeze(-1).float()
        
        return frame_embeddings