import torch
import typing
import itertools

from ._base import BaseDecoder

from src.types import ForwardOutput, DecodingStrategy, EvaluationResult

class GreedyDecoder(BaseDecoder):
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
        score_threshold: float,
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
        # (Batch Size, Prompts, Spans)
        similarity_scores = torch.sigmoid(forward_output.similarity_matrix)
        
        # (Batch Size, Spans, 2)
        candidate_spans = forward_output.candidate_spans_indices.cpu().numpy()
        
        # (Batch Size, Prompts)
        prompts_mask = forward_output.prompts_mask.cpu().numpy()
        # (Batch Size, Spans)
        spans_mask = forward_output.candidate_spans_mask.cpu().numpy()

        batch_size = similarity_scores.shape[0]
        batch_results = []

        for batch_index in range(batch_size):
            # NOTE: get all predictions above the threshold
            potential_predictions = []
            num_prompts = int(prompts_mask[batch_index].sum())
            num_spans = int(spans_mask[batch_index].sum())

            for prompt_index, span_index in itertools.product(range(num_prompts), range(num_spans)):
                score = similarity_scores[batch_index, prompt_index, span_index].item()
                if score > score_threshold:
                    start, end = candidate_spans[batch_index, span_index]
                    potential_predictions.append({
                        "score": score,
                        "prompt_idx": prompt_index,
                        "span": (start, end),
                    })

            # NOTE: sort predictions by score (descending) for greedy selection
            potential_predictions.sort(key=lambda x: x["score"], reverse=True)

            final_predictions = []
            if self.strategy == DecodingStrategy.OVERLAP:
                final_predictions = potential_predictions
            else:
                selected_spans = []
                for prediction in potential_predictions:
                    current_span = prediction["span"]
                    is_valid = True
                    for other_span in selected_spans:
                        # NOTE: we check for overlap
                        start_1, end_1 = current_span
                        start_2, end_2 = other_span
                        
                        # NOTE: partial overlap condition: one starts before the other ends, but they are not perfectly nested.
                        is_overlapping = max(start_1, start_2) <= min(end_1, end_2)
                        is_fully_nested = (start_1 >= start_2 and end_1 <= end_2) or (start_2 >= start_1 and end_2 <= end_1)
                        
                        if is_overlapping:
                            # NOTE: reject as no overlap is allowed on FLAT version
                            if self.strategy == DecodingStrategy.FLAT:
                                is_valid = False
                                break
                            # NOTE: for NESTER we only allow fully nested spans
                            elif self.strategy == DecodingStrategy.NESTED:
                                if not is_fully_nested:
                                    is_valid = False
                                    break
                    
                    if is_valid:
                        final_predictions.append(prediction)
                        selected_spans.append(current_span)
            
            motion_length = forward_output.candidate_spans_mask.shape[1]
            results_for_sample = []
            for prediction in final_predictions:
                results_for_sample.append((
                    prompts[prediction["prompt_idx"]],
                    prediction["span"][0],
                    prediction["span"][1],
                    prediction["score"]
                ))

            batch_results.append(EvaluationResult(
                motion_length=motion_length,
                predictions=results_for_sample
            ))

        return batch_results
