import torch
import typing

from ._base import BaseMotionFramesEncoder

HML3D_FEATURES_SIZE = 263

class MLPMotionFramesEncoder(BaseMotionFramesEncoder):
    def __init__(
        self,
        dropout: float,
        hidden_dim: int,
        pretrained: bool,
        frozen: bool,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.frozen_ = frozen
        self.pretrained_ = pretrained
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(HML3D_FEATURES_SIZE, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        if frozen:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        motion_features: torch.Tensor,
        motion_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None
    ) -> torch.Tensor:
        batch_size, seq_len, feat_dim = motion_features.shape
        
        reshaped_features = motion_features.view(-1, feat_dim)
        
        encoded_features = self.mlp(reshaped_features)
        
        encoded_features = encoded_features.view(batch_size, seq_len, -1)
        
        encoded_features = encoded_features * motion_masks.unsqueeze(-1).float()
        
        return encoded_features
    
    @property
    def pretrained(self) -> bool:
        return False

    @property
    def frozen(self) -> bool:
        return False
