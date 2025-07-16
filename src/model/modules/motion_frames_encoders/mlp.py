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
        activation: str = "relu",
        frozen: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.frozen = frozen
        
        layers = []
        
        layers.append(torch.nn.Linear(input_dim, hidden_dim))
        if activation == "relu":
            layers.append(torch.nn.ReLU())
        elif activation == "gelu":
            layers.append(torch.nn.GELU())
        elif activation == "tanh":
            layers.append(torch.nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            if activation == "relu":
                layers.append(torch.nn.ReLU())
            elif activation == "gelu":
                layers.append(torch.nn.GELU())
            elif activation == "tanh":
                layers.append(torch.nn.Tanh())
        
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
