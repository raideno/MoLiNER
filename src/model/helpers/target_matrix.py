import torch
import typing

from src.types import ForwardOutput, ProcessedBatch

def create_target_matrix(
    forward_output: ForwardOutput, 
    batch: ProcessedBatch
) -> typing.Tuple[torch.Tensor, int]:
    """
    Creates the target matrix for training by matching ground truth spans with candidate spans using exact frame matching.
    Create a binary target matrix with the same shape as the model output logits (B, P, S).
    target_matrix[b, p, s] = 1 if candidate_spans[b, s] matches any of the target spans for prompt p in batch b,
    
    Returns:
        torch.Tensor: (Batch Size, #Prompts, #Spans) binary target matrix
    """
    if batch.target_spans is None or batch.target_spans_per_prompt_mask is None:
        raise ValueError("Batch must contain target spans and target spans mask to create the target matrix.")
    
    # (Batch Size, #Spans, 2)
    candidate_spans = forward_output.candidate_spans_indices
    
    # NOTE: (Batch Size, #Prompts, #Spans, 2)
    groundtruth_spans = batch.target_spans
    # NOTE: (Batch Size, #Prompts, #Spans)
    groundtruth_spans_mask = batch.target_spans_per_prompt_mask
    
    # Reshape for broadcasting
    # NOTE: (Batch Size, #Prompts,  #GroundTruthSpans,  1,                  2)
    groundtruth_spans_expanded = groundtruth_spans.unsqueeze(3)
    # NOTE: (Batch Size, 1,         1,                  #CandidateSpans,    2)
    candidate_spans_expanded = candidate_spans.unsqueeze(1).unsqueeze(1)
    
    # NOTE: (Batch Size, #Prompts, #GroundTruthSpans, #CandidateSpans)
    # A match is a match only if both start and end frames match exactly
    # Boolean tensor indicating if each ground truth span matches any candidate span
    matches = torch.all(groundtruth_spans_expanded == candidate_spans_expanded, dim=-1)
    
    # NOTE: (Batch Size, #Prompts, #Spans, 1)
    temp = groundtruth_spans_mask.unsqueeze(-1)
    # NOTE: (Batch Size, #Prompts, #Spans, #Spans)
    # We might have matches between groundtruth spans and candidate spans that are padding spans, so we ignore them
    matches = matches * temp
    
    # NOTE: (Batch Size, #Prompts, #Spans)
    # For a given prompt and candidate, target is 1 if the candidate matches ANY of the ground-truth spans.
    target_matrix = torch.any(matches, dim=2).float()
    
    # --- --- --- xxx --- --- ---
    
    # TODO: write tests to check that the count is correct
    total_groundtruth_spans = int(groundtruth_spans_mask.sum().item())
    matched_candidate_spans = int(target_matrix.sum().item())
    num_unmatched_gt_spans = total_groundtruth_spans - matched_candidate_spans
    
    # --- --- --- xxx --- --- ---
    
    return target_matrix, num_unmatched_gt_spans