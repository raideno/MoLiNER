import torch
import typing

def create_projection_layer(hidden_size: int, dropout: float, out_dim: typing.Optional[int] = None) -> torch.nn.Sequential:
    """
    Creates a projection layer with specified configurations.
    """
    if out_dim is None:
        out_dim = hidden_size

    return torch.nn.Sequential(
        torch.nn.Linear(hidden_size, out_dim * 4),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(out_dim * 4, out_dim)
    )