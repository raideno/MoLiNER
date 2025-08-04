import enum
import torch
import typing

import numpy as np

from src.types import (
    MolinerForwardOutput,
    EvaluationResult,
    RawBatch,
)

from ._base import BaseDecoder

class DecodingStrategy(enum.Enum):
    """
    Defines the strategy for handling overlapping spans during decoding.

    - FLAT: No overlaps are allowed. The highest-scoring non-overlapping spans are chosen.
    - NESTED: Allows spans that are fully nested within another selected span, but prohibits partial overlaps.
    - OVERLAP: Allows any overlap. All spans above the score threshold are selected.
    """
    FLAT = "flat"
    NESTED = "nested"
    OVERLAP = "overlap"

class GreedyDecoder(BaseDecoder):
    def __init__(
        self,
        strategy: DecodingStrategy
    ):
        super().__init__()
        self.strategy = strategy

    def forward(
        self,
        forward_output: MolinerForwardOutput,
        batch: RawBatch,
        score_threshold: float,
    ) -> EvaluationResult:
        """
        Decode the forward output to produce evaluation results.
        
        Args:
            forward_output: The output from the model's forward pass
            batch: The raw batch containing original prompts information
            score_threshold: Minimum score threshold for predictions
            
        Returns:
            List of EvaluationResult objects, one per motion in the batch
        """
        similarity_scores = torch.sigmoid(forward_output.similarity_matrix)
        
        prompts_mask = forward_output.prompts_mask.detach().cpu().numpy()
        spans_mask = forward_output.candidate_spans_mask.detach().cpu().numpy()
        candidate_spans = forward_output.candidate_spans_indices.detach().cpu().numpy()
        
        batch_size = similarity_scores.shape[0]
        batch_motion_lengths = []
        batch_predictions = []

        for batch_index in range(batch_size):
            batch_scores = similarity_scores[batch_index].detach().cpu().numpy()
            num_prompts = int(prompts_mask[batch_index].sum())
            num_spans = int(spans_mask[batch_index].sum())
            
            motion_length = int(batch.motion_mask[batch_index].sum().item())
            batch_motion_lengths.append(motion_length)
            
            valid_scores = batch_scores[:num_prompts, :num_spans]
            valid_spans = candidate_spans[batch_index, :num_spans]
            
            # NOTE: keep only above threshold predictions
            above_threshold = valid_scores > score_threshold
            prompt_indices, span_indices = np.where(above_threshold)
            
            if len(prompt_indices) == 0:
                batch_predictions.append([])
                continue
            
            scores = valid_scores[prompt_indices, span_indices]
            
            predictions = np.empty(len(scores), dtype=[
                ('score', 'f4'),
                ('prompt_idx', 'i4'), 
                ('span_start', 'i4'),
                ('span_end', 'i4')
            ])
            
            predictions['score'] = scores
            predictions['prompt_idx'] = prompt_indices
            predictions['span_start'] = valid_spans[span_indices, 0]
            predictions['span_end'] = valid_spans[span_indices, 1]
            
            predictions = np.sort(predictions, order='score')[::-1]
            
            final_predictions = self._apply_strategy(predictions)
            
            motion_predictions = []
            batch_prompts = batch.prompts[batch_index] if batch_index < len(batch.prompts) else []
            
            for pred in final_predictions:
                prompt_idx = int(pred['prompt_idx'])
                if prompt_idx < len(batch_prompts):
                    # NOTE: extract text from tuple
                    prompt_text = batch_prompts[prompt_idx][0]
                    span_tuple = (int(pred['span_start']), int(pred['span_end']))
                    score = float(pred['score'])
                    
                    motion_predictions.append((prompt_text, span_tuple, score))
            
            batch_predictions.append(motion_predictions)
            
        # NOTE: group predictions by prompt text for each motion
        grouped_batch_predictions = []
        for motion_predictions in batch_predictions:
            prompt_dict = {}
            for prompt_text, span_tuple, score in motion_predictions:
                if prompt_text not in prompt_dict:
                    prompt_dict[prompt_text] = []
                prompt_dict[prompt_text].append((span_tuple[0], span_tuple[1], score))
            grouped_predictions = [
                (prompt, spans_scores)
                for prompt, spans_scores in prompt_dict.items()
            ]
            grouped_batch_predictions.append(grouped_predictions)

        return EvaluationResult(
            motion_length=batch_motion_lengths,
            predictions=grouped_batch_predictions
        )
    
    def _apply_strategy(self, predictions):
        if self.strategy == DecodingStrategy.OVERLAP:
            return predictions
        
        if len(predictions) <= 1:
            return predictions
        
        spans = np.column_stack([predictions['span_start'], predictions['span_end']])
        
        if self.strategy == DecodingStrategy.FLAT:
            return self._apply_flat_strategy(predictions, spans)
        elif self.strategy == DecodingStrategy.NESTED:
            return self._apply_nested_strategy(predictions, spans)
        
        return predictions
    
    def _apply_flat_strategy(self, predictions, spans):
        """
        No overlaps allowed.
        """
        if len(predictions) <= 1:
            return predictions
            
        selected_mask = np.zeros(len(predictions), dtype=bool)
        selected_mask[0] = True
        selected_spans = [spans[0]]
        
        for i in range(1, len(predictions)):
            current_span = spans[i]
            
            selected_spans_array = np.array(selected_spans)
            
            # NOTE: check overlap: max(start1, start2) <= min(end1, end2)
            max_starts = np.maximum(current_span[0], selected_spans_array[:, 0])
            min_ends = np.minimum(current_span[1], selected_spans_array[:, 1])
            overlaps = max_starts <= min_ends
            
            if not np.any(overlaps):
                selected_mask[i] = True
                selected_spans.append(current_span)
        
        return predictions[selected_mask]
    
    def _apply_nested_strategy(self, predictions, spans):
        """
        Fully nested spans are allowed.
        """
        if len(predictions) <= 1:
            return predictions
            
        selected_mask = np.zeros(len(predictions), dtype=bool)
        selected_mask[0] = True
        selected_spans = [spans[0]]
        
        for i in range(1, len(predictions)):
            current_span = spans[i]
            is_valid = True
            
            for selected_span in selected_spans:
                start_1, end_1 = current_span
                start_2, end_2 = selected_span
                
                # NOTE: check for overlap
                is_overlapping = max(start_1, start_2) <= min(end_1, end_2)
                
                if is_overlapping:
                    # NOTE: check if fully nested
                    is_fully_nested = (
                        (start_1 >= start_2 and end_1 <= end_2) or 
                        (start_2 >= start_1 and end_2 <= end_1)
                    )
                    
                    if not is_fully_nested:
                        is_valid = False
                        break
            
            if is_valid:
                selected_mask[i] = True
                selected_spans.append(current_span)
        
        return predictions[selected_mask]
