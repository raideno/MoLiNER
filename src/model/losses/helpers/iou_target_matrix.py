import torch
import typing

from src.types import ForwardOutput, ProcessedBatch

def compute_iou_spans(
    spans1: torch.Tensor, 
    spans2: torch.Tensor
) -> torch.Tensor:
    """
    Computes IoU (Intersection over Union) between two sets of spans.
    
    Args:
        spans1: Tensor of shape (..., 2) containing [start, end] frames
        spans2: Tensor of shape (..., 2) containing [start, end] frames
    
    Returns:
        torch.Tensor: IoU values between 0 and 1
    """
    start1, end1 = spans1[..., 0], spans1[..., 1]
    start2, end2 = spans2[..., 0], spans2[..., 1]
    
    intersection_start = torch.max(start1, start2)
    intersection_end = torch.min(end1, end2)
    intersection = torch.clamp(intersection_end - intersection_start + 1, min=0)
    
    union_start = torch.min(start1, start2)
    union_end = torch.max(end1, end2)
    union = union_end - union_start + 1
    
    iou = intersection / torch.clamp(union, min=1e-8)
    
    return iou

def create_iou_target_matrix(
    forward_output: ForwardOutput, 
    batch: ProcessedBatch,
    iou_threshold: float = 0.5
) -> typing.Tuple[torch.Tensor, int]:
    """
    Creates a target matrix using IoU-based matching instead of exact frame matching.
    For each prompt-candidate pair, computes the maximum IoU with any ground truth span.
    
    Args:
        forward_output: Model forward output containing candidate spans
        batch: Processed batch containing ground truth spans
        iou_threshold: Minimum IoU threshold to consider a match (for counting unmatched spans)
    
    Returns:
        torch.Tensor: (Batch Size, #Prompts, #Spans) IoU values between candidates and ground truth
        int: Number of ground truth spans that couldn't be matched above threshold
    """
    if batch.target_spans is None or batch.target_spans_per_prompt_mask is None:
        raise ValueError("Batch must contain target spans and target spans mask to create the IoU target matrix.")
    
    # (Batch Size, #Spans, 2)
    candidate_spans = forward_output.candidate_spans_indices
    
    # NOTE: (Batch Size, #Prompts, #GroundTruthSpans, 2)
    groundtruth_spans = batch.target_spans
    # NOTE: (Batch Size, #Prompts, #GroundTruthSpans)
    groundtruth_spans_mask = batch.target_spans_per_prompt_mask
    
    batch_size, num_prompts, max_gt_spans, _ = groundtruth_spans.shape
    _, num_candidate_spans, _ = candidate_spans.shape
    
    # Reshape for broadcasting
    # NOTE: (Batch Size, #Prompts, #GroundTruthSpans, 1, 2)
    groundtruth_spans_expanded = groundtruth_spans.unsqueeze(3)
    # NOTE: (Batch Size, 1, 1, #CandidateSpans, 2)
    candidate_spans_expanded = candidate_spans.unsqueeze(1).unsqueeze(1)
    
    # NOTE: (Batch Size, #Prompts, #GroundTruthSpans, #CandidateSpans)
    # Compute IoU between all ground truth and candidate span pairs
    iou_matrix = compute_iou_spans(groundtruth_spans_expanded, candidate_spans_expanded)
    
    # NOTE: (Batch Size, #Prompts, #GroundTruthSpans, 1)
    # Mask out invalid ground truth spans (padding)
    gt_mask_expanded = groundtruth_spans_mask.unsqueeze(-1)
    # NOTE: (Batch Size, #Prompts, #GroundTruthSpans, #CandidateSpans)
    iou_matrix = iou_matrix * gt_mask_expanded
    
    # NOTE: (Batch Size, #Prompts, #CandidateSpans)
    # For each prompt-candidate pair, take the maximum IoU with any ground truth span
    target_matrix, _ = torch.max(iou_matrix, dim=2)
    
    # --- Count unmatched spans for monitoring ---
    
    # For each ground truth span, find the maximum IoU with any candidate
    max_iou_per_gt, _ = torch.max(iou_matrix, dim=3)  # (Batch Size, #Prompts, #GroundTruthSpans)
    
    # Count ground truth spans that don't have a candidate with IoU >= threshold
    matched_gt_spans = (max_iou_per_gt >= iou_threshold) * groundtruth_spans_mask
    total_gt_spans = int(groundtruth_spans_mask.sum().item())
    matched_count = int(matched_gt_spans.sum().item())
    unmatched_spans_count = total_gt_spans - matched_count
    
    return target_matrix, unmatched_spans_count
