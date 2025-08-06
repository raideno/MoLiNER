import torch
import torch.nn as nn
import typing

from ._base import BaseMotionEncoder

class MLPMotionEncoder(BaseMotionEncoder):
    def __init__(
        self,
        frozen: bool,
        pretrained: bool,
        num_frames: int,
        dropout: float = 0.1,
        input_dim: int = 263,
        output_dim: int = 256,
    ):
        """
        Args:
            num_frames (int): Fixed number of frames in the motion sequence
            input_dim (int): Dimension of input motion features per frame
            output_dim (int): Output dimension for encoded features
        """
        super().__init__()
        
        self.dropout = dropout
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        flattened_input_dim = num_frames * input_dim
        final_output_dim = output_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(flattened_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, final_output_dim)
        )
        
    def forward(
        self,
        motion_features: torch.Tensor,
        motion_masks: torch.Tensor,
        batch_index: typing.Optional[int] = None
    ) -> torch.Tensor:
        batch_size, seq_len, input_dim = motion_features.shape
        
        if seq_len != self.num_frames:
            raise ValueError(
                f"Expected sequence length {self.num_frames}, got {seq_len}. "
                "MLP encoder requires fixed sequence length."
            )
        
        if input_dim != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {input_dim}."
            )
        
        # NOTE: (batch_size, num_frames * input_dim)
        flattened_input = motion_features.view(batch_size, -1)
        
        # NOTE: (batch_size, output_dim)
        mlp_output = self.mlp(flattened_input)

        return mlp_output
        
    @property
    def pretrained(self) -> bool:
        return False

    @property
    def frozen(self) -> bool:
        return False