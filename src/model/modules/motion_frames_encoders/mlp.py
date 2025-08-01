import torch
import typing

from ._base import BaseMotionFramesEncoder

class MLPMotionFramesEncoder(BaseMotionFramesEncoder):
    def __init__(
        self,
        input_dim: int = 263,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        pretrained: bool = False,
        frozen: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.frozen_ = frozen
        self.pretrained_ = pretrained
        
        layers: typing.List[torch.nn.Module] = []
        
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
        
        if num_layers > 1:
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        
        self.mlp = torch.nn.Sequential(*layers)
        
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