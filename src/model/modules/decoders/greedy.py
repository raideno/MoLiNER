import torch
import typing

import numpy as np

from src.types import ForwardOutput, DecodingStrategy, EvaluationResult

from ._base import BaseDecoder

class GreedyDecoder(BaseDecoder):
    def __init__(
        self,
        strategy: DecodingStrategy
    ):
        super().__init__()
        self.strategy = strategy

    def decode(
        self,
        forward_output: ForwardOutput,
        prompts: typing.List[typing.List[str]],
        score_threshold: float,
    ) -> typing.List[EvaluationResult]:
        similarity_scores = torch.sigmoid(forward_output.similarity_matrix)
        
        prompts_mask = forward_output.prompts_mask.detach().cpu().numpy()
        spans_mask = forward_output.candidate_spans_mask.detach().cpu().numpy()
        candidate_spans = forward_output.candidate_spans_indices.detach().cpu().numpy()
        
        batch_size = similarity_scores.shape[0]
        batch_results = []

        for batch_index in range(batch_size):
            batch_scores = similarity_scores[batch_index].detach().cpu().numpy()
            num_prompts = int(prompts_mask[batch_index].sum())
            num_spans = int(spans_mask[batch_index].sum())
            
            valid_scores = batch_scores[:num_prompts, :num_spans]
            valid_spans = candidate_spans[batch_index, :num_spans]
            
            above_threshold = valid_scores > score_threshold
            
            prompt_indices, span_indices = np.where(above_threshold)
            
            if len(prompt_indices) == 0:
                motion_length = forward_output.candidate_spans_mask.shape[1]
                batch_results.append(EvaluationResult(
                    motion_length=motion_length,
                    predictions=[]
                ))
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
            
            motion_length = forward_output.candidate_spans_mask.shape[1]
            results_for_sample = []
            batch_prompts = prompts[batch_index] if batch_index < len(prompts) else []
            
            for pred in final_predictions:
                prompt_idx = int(pred['prompt_idx'])
                if prompt_idx < len(batch_prompts):
                    results_for_sample.append((
                        batch_prompts[prompt_idx],
                        int(pred['span_start']),
                        int(pred['span_end']),
                        float(pred['score'])
                    ))

            batch_results.append(EvaluationResult(
                motion_length=motion_length,
                predictions=results_for_sample
            ))

        return batch_results
    
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