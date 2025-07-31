import torch
import typing

from ._base import BaseMotionFramesEncoder

class LSTMMotionFramesEncoder(BaseMotionFramesEncoder):
    def __init__(
        self,
        input_dim: int = 263,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        pretrained: bool = False,
        frozen: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.frozen_ = frozen
        self.pretrained_ = pretrained
        
        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_projection = torch.nn.Linear(lstm_output_dim, hidden_dim)
        
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
        
        lengths = motion_masks.sum(dim=1).cpu()
        
        # NOTE: this is a sort of collate function for variable-length sequences for RNNs
        # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            motion_features,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        packed_output, _ = self.lstm(packed_input)
        
        unpacked_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=seq_len
        )
        
        projected_output = self.output_projection(unpacked_output)
        
        projected_output = projected_output * motion_masks.unsqueeze(-1).float()
        
        return projected_output
    
    @property
    def pretrained(self) -> bool:
        return False

    @property
    def frozen(self) -> bool:
        return False
