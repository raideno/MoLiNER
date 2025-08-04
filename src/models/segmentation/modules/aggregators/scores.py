import torch
import typing

from ._base import BaseAggregator

class ScoresAggregator(BaseAggregator):
    """
    Aggregates window-level predictions by accumulating scores for each frame
    and selecting the class with the highest accumulated score.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        motion_length: int,
        window_metadata: torch.Tensor,
        class_probs: torch.Tensor,
        threshold: float,
        batch_index: typing.Optional[int] = None
    ) -> torch.Tensor:
        """
        Vote for frame-level class predictions by accumulating scores.
        
        Args:
            motion_length: Length of the motion sequence
            window_metadata: Window positions [num_windows, 3] -> [batch_idx, start_frame, end_frame]
            class_probs: Class probabilities [num_windows, num_classes]
            threshold: Confidence threshold
            batch_index: Optional batch index for logging/debugging
            
        Returns:
            Tensor of shape [motion_length] with class indices (-1 for no prediction)
        """
        num_windows = window_metadata.shape[0]
        num_classes = class_probs.shape[1]
        
        # NOTE: (motion_length, num_classes)
        frame_class_scores = torch.zeros(motion_length, num_classes, device=class_probs.device)
        
        for window_idx in range(num_windows):
            window_start = window_metadata[window_idx, 1].item()
            window_end = window_metadata[window_idx, 2].item()
            
            # NOTE: (num_classes,)
            window_probs = class_probs[window_idx]
            
            for frame_idx in range(window_start, min(window_end + 1, motion_length)):
                frame_class_scores[frame_idx] += window_probs
        
        # NOTE: class with highest score for each frame
        max_scores, predicted_classes = torch.max(frame_class_scores, dim=1)
        
        # NOTE: model not confident enough
        predicted_classes[max_scores < threshold] = -1
        
        return predicted_classes
