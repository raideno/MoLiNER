import torch
import typing

from src.types import EvaluationResult

DEFAULT_CONFIDENCE_THRESHOLD = 0.5

def segmentation_predictions_to_evaluation_result(
        predictions: typing.List[torch.Tensor],
        class_names: typing.Optional[typing.List[str]] = None,
    ) -> EvaluationResult:
        """
        Convert frame-level class predictions to EvaluationResult format.
        
        Args:
            predictions: List of tensors from predict() method, each tensor has shape [motion_length]
                        with class indices (-1 for no prediction above threshold)
            class_names: Optional list of class names. If None, uses "class_0", "class_1", etc.
            
        Returns:
            EvaluationResult with motion lengths and predictions converted to span format
        """
        motion_lengths = [len(pred) for pred in predictions]
        
        batch_predictions = []
        
        for motion_idx, frame_predictions in enumerate(predictions):
            motion_predictions = []
            
            current_class: typing.Optional[int] = None
            current_span_start: typing.Optional[int] = None
            
            for frame_idx, class_tensor in enumerate(frame_predictions):
                class_idx = int(class_tensor.item())
                
                if class_idx == -1:
                    if current_class is not None and current_span_start is not None:
                        # NOTE: end current span
                        class_name = class_names[current_class] if class_names else f"class_{current_class}"
                        # TODO: instead we should use the average of the frames scores
                        span = (current_span_start, frame_idx - 1, DEFAULT_CONFIDENCE_THRESHOLD)
                        motion_predictions.append((class_name, [span]))
                        current_class = None
                        current_span_start = None
                elif class_idx != current_class:
                    # NOTE: class changed
                    if current_class is not None and current_span_start is not None:
                        # NOTE: end previous span
                        class_name = class_names[current_class] if class_names else f"class_{current_class}"
                        # TODO: instead we should use the average of the frames scores
                        span = (current_span_start, frame_idx - 1, DEFAULT_CONFIDENCE_THRESHOLD)
                        motion_predictions.append((class_name, [span]))
                    
                    # NOTE: start new span
                    current_class = class_idx
                    current_span_start = frame_idx
            
            if current_class is not None and current_span_start is not None:
                class_name = class_names[current_class] if class_names else f"class_{current_class}"
                # TODO: instead we should use the average of the frames scores
                span = (current_span_start, len(frame_predictions) - 1, DEFAULT_CONFIDENCE_THRESHOLD)
                motion_predictions.append((class_name, [span]))
            
            batch_predictions.append(motion_predictions)
        
        return EvaluationResult(
            motion_length=motion_lengths,
            predictions=batch_predictions
        )
