import torch
import typing

from src.types import SegmenterForwardOutput, RawBatch
from src.constants import LOCATE_CLASSES, __get_locate_classes_flat_list

from ._base import BaseLoss

NO_CLASS_INDEX = -1

class StandardLoss(BaseLoss):
    def __init__(self):
        super().__init__()
        
        self.classification_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=NO_CLASS_INDEX)
        self.start_regression_loss_fn = torch.nn.MSELoss()
        self.end_regression_loss_fn = torch.nn.MSELoss()
    
    def forward(
        self,
        forward_output: SegmenterForwardOutput,
        batch: RawBatch,
        batch_index: typing.Optional[int] = None,
    ) -> torch.Tensor:
        # NOTE: (total_windows, num_classes)
        class_logits = forward_output.class_logits
        # NOTE: (total_windows, 1)
        start_logits = forward_output.start_logits
        # NOTE: (total_windows, 1)
        end_logits = forward_output.end_logits
        
        start_logits = start_logits.view(-1)
        end_logits = end_logits.view(-1)
        
        # NOTE: [total_windows, 3]
        labels = extract_window_labels(batch, forward_output)
        
        class_targets = labels[:, 0].long()
        class_loss = self.classification_loss_fn(class_logits, class_targets)
        
        valid_windows = class_targets != -1
        
        if valid_windows.any():
            valid_start_mask = (labels[:, 1] != -1) & valid_windows
            valid_end_mask = (labels[:, 2] != -1) & valid_windows
            
            if valid_start_mask.any():
                start_loss = self.start_regression_loss_fn(
                    start_logits[valid_start_mask], 
                    labels[:, 1][valid_start_mask]
                )
            else:
                start_loss = torch.tensor(0.0, device=labels.device)
                
            if valid_end_mask.any():
                end_loss = self.end_regression_loss_fn(
                    end_logits[valid_end_mask], 
                    labels[:, 2][valid_end_mask]
                )
            else:
                end_loss = torch.tensor(0.0, device=labels.device)
        else:
            start_loss = torch.tensor(0.0, device=labels.device)
            end_loss = torch.tensor(0.0, device=labels.device)
        
        total_loss = class_loss + start_loss + end_loss
        
        return total_loss

def extract_window_labels(
    batch: RawBatch, 
    forward_output: SegmenterForwardOutput
) -> torch.Tensor:
    """
    Extract multi-class labels for each window.
    
    Returns:
        labels: Tensor of shape [total_windows, 3]
                labels[:, 0] = class index (-1 for no class, 0-19 for valid classes)
                labels[:, 1] = relative start position (0-1 or -1 if no transition)
                labels[:, 2] = relative end position (0-1 or -1 if no transition)
    """
    total_windows = forward_output.class_logits.shape[0]
    # [total_windows, 3] -> [batch_idx, start_frame, end_frame]
    window_metadata = forward_output.windows_positions
    device = forward_output.class_logits.device
    
    labels = torch.full((total_windows, 3), NO_CLASS_INDEX, device=device)
    
    for window_idx in range(total_windows):
        batch_idx = window_metadata[window_idx, 0].item()
        window_start_frame = window_metadata[window_idx, 1].item()
        window_end_frame = window_metadata[window_idx, 2].item()
        window_length = window_end_frame - window_start_frame + 1
        
        if batch_idx < len(batch.prompts):
            sample_prompts = batch.prompts[batch_idx]
            
            overlapping_spans = []
            
            for prompt in sample_prompts:
                text, spans, is_sequence_prompt = prompt
                
                class_idx = extract_class_from_text(text)
                
                if class_idx != -1:
                    for span_start, span_end in spans:
                        if (span_start <= window_end_frame and span_end >= window_start_frame):
                            overlapping_spans.append((class_idx, span_start, span_end))
            
            if overlapping_spans:
                if len(overlapping_spans) > 1:
                    winning_class = resolve_overlapping_spans(overlapping_spans)
                else:
                    winning_class = overlapping_spans[0][0]
                
                winning_spans = [span for span in overlapping_spans if span[0] == winning_class]
                
                all_starts = [max(span[1], window_start_frame) for span in winning_spans]
                all_ends = [min(span[2], window_end_frame) for span in winning_spans]
                
                overall_start = min(all_starts)
                overall_end = max(all_ends)
                
                rel_start = (overall_start - window_start_frame) / (window_length - 1) if window_length > 1 else 0.0
                rel_end = (overall_end - window_start_frame) / (window_length - 1) if window_length > 1 else 0.0
                
                labels[window_idx, 0] = float(winning_class)
                labels[window_idx, 1] = rel_start
                labels[window_idx, 2] = rel_end
                
            # NOTE: if no overlapping spans with current window, labels remain -1 (no class)
    
    return labels

def extract_class_from_text(text: str) -> int:
    """
    Extract class index from prompt text by matching with LOCATE_CLASSES.
    """
    text_lower = text.lower().strip()
    
    for i, class_term in enumerate(LOCATE_CLASSES):
        if class_term.lower() in text_lower:
            return i
    
    # TODO: maybe we should perform a similarity using some Bert like model to assign the closest class if above a threshold.
    return NO_CLASS_INDEX

def resolve_overlapping_spans(overlapping_spans: typing.List[typing.Tuple[int, int, int]]) -> int:
    """
    Args:
        overlapping_spans: List of (class_idx, span_start, span_end) tuples
        
    Returns:
        The winning class index (most frequent class)
    """
    if not overlapping_spans:
        return NO_CLASS_INDEX

    class_counts: typing.Dict[int, int] = {}
    for class_idx, _, _ in overlapping_spans:
        if class_idx in class_counts:
            class_counts[class_idx] += 1
        else:
            class_counts[class_idx] = 1

    return max(class_counts, key=lambda k: class_counts[k], default=NO_CLASS_INDEX)
