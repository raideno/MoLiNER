import torch
import warnings

def reduce(
    logits: torch.Tensor,
    reduction: str = "mean",
):
    if reduction == "none":
        loss = logits
    elif reduction == "mean":
        # loss = loss.sum() / valid_mask.sum()  # Normalize by the number of valid (non-ignored) elements
        loss = logits.mean()
    elif reduction == 'sum':
        loss = logits.sum()
    else:
        warnings.warn(
            f"Invalid Value for config 'loss_reduction': '{reduction} \n Supported reduction modes:"
            f" 'none', 'mean', 'sum'. It will be used 'sum' instead.")
        loss = logits.sum()
    
    return loss