import torch
import typing
import warnings

def reduce(
    logits: torch.Tensor,
    reduction: str = "mean",
    valid_mask: typing.Optional[torch.Tensor] = None
):
    if reduction == "none":
        loss = logits
    elif reduction == "mean":
        if valid_mask is not None:
            # NOTE: normalize by the number of valid (non-ignored) elements
            loss = logits.sum() / valid_mask.sum()
        else:
            raise ValueError("valid_mask must be provided for 'mean' reduction.")
    elif reduction == 'sum':
        loss = logits.sum()
    else:
        warnings.warn(
            f"Invalid Value for config 'loss_reduction': '{reduction} \n Supported reduction modes:"
            f" 'none', 'mean', 'sum'. It will be used 'sum' instead.")
        loss = logits.sum()
    
    return loss