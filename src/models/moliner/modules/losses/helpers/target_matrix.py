import torch
import typing

from src.types import MolinerForwardOutput, RawBatch

def compute_span_iou(span_1: torch.Tensor, span_2: torch.Tensor) -> torch.Tensor:
    start1, end1 = span_1[..., 0], span_1[..., 1]
    start2, end2 = span_2[..., 0], span_2[..., 1]
    
    intersection_start = torch.max(start1, start2)
    intersection_end = torch.min(end1, end2)
    intersection = torch.clamp(intersection_end - intersection_start, min=0)
    
    union = (end1 - start1) + (end2 - start2) - intersection
    
    # NOTE: to not divide by zero
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
    
    return iou

def create_target_matrix(
    forward_output: MolinerForwardOutput, 
    batch: RawBatch,
    threshold: float
) -> torch.Tensor:
    """
    Create a target matrix with the same shape as the model output logits (B, P, S).
    target_matrix[b, p, s] = IoU if IoU between candidate_spans[b, s] and any target span for prompt p 
    in batch b is above threshold, otherwise 0.
    
    Args:
        forward_output: Model forward output containing candidate spans
        batch: Processed batch containing ground truth spans
        threshold: Minimum IoU threshold to consider a match (default: 0.5)
    
    Returns:
        torch.Tensor: (Batch Size, #Prompts, #Spans) target matrix with IoU values
        int: Number of unmatched ground truth spans
    """
    if batch.target_spans is None or batch.target_spans_per_prompt_mask is None:
        raise ValueError("Batch must contain target spans and target spans mask to create the target matrix.")
    
    # (Batch Size, #Spans, 2)
    candidate_spans = forward_output.candidate_spans_indices
    
    # NOTE: (Batch Size, #Prompts, #Spans, 2)
    groundtruth_spans = batch.target_spans
    # NOTE: (Batch Size, #Prompts, #Spans)
    groundtruth_spans_mask = batch.target_spans_per_prompt_mask
    
    # NOTE: (Batch Size, #Prompts,  #GroundTruthSpans,  1,                  2)
    groundtruth_spans_expanded = groundtruth_spans.unsqueeze(3)
    # NOTE: (Batch Size, 1,         1,                  #CandidateSpans,    2)
    candidate_spans_expanded = candidate_spans.unsqueeze(1).unsqueeze(1)
    
    # NOTE: (Batch Size, #Prompts, #GroundTruthSpans, #CandidateSpans)
    # Compute IoU between every ground truth span and every candidate span
    iou_matrix = compute_span_iou(groundtruth_spans_expanded, candidate_spans_expanded)
    
    # NOTE: (Batch Size, #Prompts, #GroundTruthSpans, 1)
    mask = groundtruth_spans_mask.unsqueeze(-1)
    # NOTE: (Batch Size, #Prompts, #GroundTruthSpans, #CandidateSpans)
    # We might have IoU values between groundtruth spans and candidate spans that are padding spans, so we ignore them
    iou_matrix = iou_matrix * mask
    
    # NOTE: (Batch Size, #Prompts, #CandidateSpans)
    # Take maximum IoU across all ground truth spans for each candidate span, like the one have the best match
    target_matrix, _ = torch.max(iou_matrix, dim=2)
    
    target_matrix = torch.where(target_matrix >= threshold, target_matrix, torch.zeros_like(target_matrix))
    
    return target_matrix
