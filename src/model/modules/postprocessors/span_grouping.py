import torch
import typing

from src.types import EvaluationResult

from ._base import BasePostprocessor

class SpanGroupingPostprocessor(BasePostprocessor):
    """
    Groups overlapping spans and spans that are close together within a specified gap.
    """
    
    def __init__(
        self,
        max_gap: int
    ):
        """
        Args:
            max_gap (int): Maximum number of frames between spans to consider them for grouping.
        """
        super().__init__()
    
        self.max_gap = max_gap
    
    def forward(
        self,
        evaluation_results: EvaluationResult
    ) -> EvaluationResult:
        grouped_predictions = []
        
        for motion_predictions in evaluation_results.predictions:
            grouped_motion_predictions = []
            
            for prompt_text, spans in motion_predictions:
                grouped_spans = self._group_spans(spans)
                grouped_motion_predictions.append((prompt_text, grouped_spans))
            
            grouped_predictions.append(grouped_motion_predictions)
        
        return EvaluationResult(
            motion_length=evaluation_results.motion_length,
            predictions=grouped_predictions
        )
    
    def _group_spans(
        self,
        spans: typing.List[typing.Tuple[int, int, float]]
    ) -> typing.List[typing.Tuple[int, int, float]]:
        """
        Group overlapping and nearby spans.
        
        Args:
            spans: List of (start, end, score) tuples.
            
        Returns:
            List of grouped spans with merged boundaries and averaged scores.
        """
        if not spans:
            return spans
        
        # NOTE: sort spans by start frame
        sorted_spans = sorted(spans, key=lambda x: x[0])
        
        grouped = []
        
        current_start, current_end, current_score = sorted_spans[0]
        
        span_count = 1
        
        for start, end, score in sorted_spans[1:]:
            # NOTE: current span and next (one iteration is at) one can be grouped if overlap or within max_gap
            if start <= current_end + self.max_gap:
                current_end = max(current_end, end)
                current_score += score
                span_count += 1
            # NOTE: if we can't group, then we finalize current group and push it to the grouped spans list
            else:
                # TODO: this should be studied more as it isn't really accurate
                averaged_score = current_score / span_count
                grouped.append((current_start, current_end, averaged_score))
                
                # NOTE: new groups starts within the span we are iterating
                current_start, current_end, current_score = start, end, score
                span_count = 1
        
        averaged_score = current_score / span_count
        grouped.append((current_start, current_end, averaged_score))
        
        return grouped
