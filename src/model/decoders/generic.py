import torch
import typing
from itertools import product

from .index import BaseDecoder
from src.data.typing import ForwardOutput, DecodingStrategy, EvaluationResult


class GenericDecoder(BaseDecoder):
    """
    A generic decoder implementation that extracts predictions based on similarity scores
    with configurable overlap handling strategies.
    """
    
    def __init__(
        self,
        strategy: DecodingStrategy = DecodingStrategy.FLAT,
    ):
        super().__init__()
        
        self.strategy = strategy

    def decode(
        self,
        forward_output: ForwardOutput,
        prompts: typing.List[str],
        score_threshold: float = 0.5,
    ) -> typing.List[EvaluationResult]:
        """
        Decodes the model's forward pass output into a list of predicted spans.

        Args:
            forward_output (ForwardOutput): The raw output from the model's forward pass.
            prompts (typing.List[str]): The original list of prompt texts for one motion.
            score_threshold (float): The minimum similarity score to consider a span as a potential match.

        Returns:
            typing.List[EvaluationResult]: A list of EvaluationResult objects, one for each item in the batch.
        """
        # (B, P, S)
        similarity_scores = torch.sigmoid(forward_output.similarity_matrix)
        
        # (B, S, 2)
        candidate_spans = forward_output.candidate_spans_indices.cpu().numpy()
        
        # (B, P) and (B, S)
        prompts_mask = forward_output.prompts_mask.cpu().numpy()
        spans_mask = forward_output.candidate_spans_mask.cpu().numpy()

        batch_size = similarity_scores.shape[0]
        batch_results = []

        for b in range(batch_size):
            # NOTE: get all predictions above the threshold
            potential_predictions = []
            num_prompts = int(prompts_mask[b].sum())
            num_spans = int(spans_mask[b].sum())

            for p, s in product(range(num_prompts), range(num_spans)):
                score = similarity_scores[b, p, s].item()
                if score > score_threshold:
                    start, end = candidate_spans[b, s]
                    potential_predictions.append({
                        "score": score,
                        "prompt_idx": p,
                        "span": (start, end),
                    })

            # NOTE: sort predictions by score (descending) for greedy selection
            potential_predictions.sort(key=lambda x: x["score"], reverse=True)

            final_predictions = []
            if self.strategy == DecodingStrategy.OVERLAP:
                final_predictions = potential_predictions
            else:
                selected_spans = []
                for pred in potential_predictions:
                    current_span = pred["span"]
                    is_valid = True
                    for other_span in selected_spans:
                        # NOTE: we check for overlap
                        s1, e1 = current_span
                        s2, e2 = other_span
                        
                        # NOTE: partial overlap condition: one starts before the other ends, but they are not perfectly nested.
                        is_overlapping = max(s1, s2) <= min(e1, e2)
                        
                        if is_overlapping:
                            if self.strategy == DecodingStrategy.FLAT:
                                is_valid = False
                                break
                            elif self.strategy == DecodingStrategy.NESTED:
                                # NOTE: check if it's a partial overlap (not fully nested)
                                is_nested = (s1 >= s2 and e1 <= e2) or (s2 >= s1 and e2 <= e1)
                                if not is_nested:
                                    is_valid = False
                                    break
                    
                    if is_valid:
                        final_predictions.append(pred)
                        selected_spans.append(current_span)
            
            motion_length = forward_output.candidate_spans_mask.shape[1]
            results_for_sample = []
            for pred in final_predictions:
                 results_for_sample.append((
                     prompts[pred["prompt_idx"]],
                     pred["span"][0],
                     pred["span"][1],
                     pred["score"]
                 ))

            batch_results.append(EvaluationResult(
                motion_length=motion_length,
                predictions=results_for_sample
            ))

        return batch_results
