import torch
import typing

from src.types import MolinerForwardOutput, Batch

def create_loss_mask(
    forward_output: MolinerForwardOutput, 
    batch: Batch
) -> torch.Tensor:
    """
    Creates a mask for valid (prompt, span) pairs for loss computation.
    Used to ignore padding spans and prompts created during forward pass.
    
    Returns:
        torch.Tensor: (B, P, S) mask where 1 indicates valid pairs
    """
    if batch.target_spans_mask is None:
        raise ValueError("Batch must contain target spans mask to create the loss mask.")
    
    # NOTE: (B, P); valid and non padding prompts
    prompt_mask = forward_output.prompts_mask
    
    # NOTE: (B, S); valid and non padding candidate spans  
    candidate_mask = forward_output.candidate_spans_mask
    
    # NOTE: (B, P, S); we combine them to create a mask of all valid pairs that should be considered for loss computation
    final_mask = prompt_mask.unsqueeze(2) * candidate_mask.unsqueeze(1)
    
    return final_mask