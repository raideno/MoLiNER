import os
import torch
import typing
import logging
import traceback

from dataclasses import dataclass

from src.model import MoLiNER
from src.types import RawBatch, ProcessedBatch, EvaluationResult
from src.visualizations.spans import plot_evaluation_results

logger = logging.getLogger(__name__)

@dataclass
class VisualizationSample:
    sample_index: int
    threshold_results: typing.Dict[float, typing.List[typing.Tuple[str, typing.List[typing.Tuple[int, int, float]]]]]
    motion_length: int
    raw_prompts: typing.List[typing.Tuple[str, typing.List[typing.Tuple[int, int]], bool]]


@dataclass
class VisualizationResults:
    samples: typing.Dict[int, VisualizationSample]
    epoch: int
    debug_info: typing.Optional[typing.Dict] = None


class VisualizationGenerator:
    def __init__(
        self,
        score_thresholds: typing.List[float],
        debug: bool = False
    ):
        """
        Args:
            score_thresholds: List of confidence thresholds for predictions.
            debug: Whether to output debug information.
        """
        self.score_thresholds = score_thresholds
        self.debug = debug
    
    def generate_visualization_data(
        self,
        model: MoLiNER,
        raw_batch: RawBatch,
        epoch: typing.Optional[int] = None
    ) -> VisualizationResults:
        """
        Generate visualization data for a batch of samples using the model's predict function.
        
        Args:
            model: The MoLiNER model to evaluate.
            raw_batch: The batch of raw data to process.
            epoch: Current epoch number for logging.
            
        Returns:
            VisualizationResults containing all generated data.
        """
        original_mode = model.training
        model.eval()
        
        samples = {}
        debug_info = {} if self.debug else None
        
        try:
            with torch.no_grad():
                # Get predictions for all thresholds using the model's predict function
                threshold_results_batch = {}
                for threshold in self.score_thresholds:
                    evaluation_result = model.predict(
                        raw_batch=raw_batch.to(model.device),
                        threshold=threshold
                    )
                    threshold_results_batch[threshold] = evaluation_result.predictions
                
                # Extract results for each sample
                for i in range(len(raw_batch.sid)):
                    sample_threshold_results = {}
                    for threshold in self.score_thresholds:
                        # Get predictions for this sample
                        sample_predictions = (
                            threshold_results_batch[threshold][i] 
                            if i < len(threshold_results_batch[threshold])
                            else []
                        )
                        sample_threshold_results[threshold] = sample_predictions
                    
                    motion_length = int(raw_batch.motion_mask[i].sum())
                    
                    samples[i] = VisualizationSample(
                        sample_index=i,
                        threshold_results=sample_threshold_results,
                        motion_length=motion_length,
                        raw_prompts=raw_batch.prompts[i]
                    )
                    
                    if self.debug and debug_info is not None:
                        debug_info[i] = self._generate_debug_info(
                            sample_data=samples[i],
                            epoch=epoch
                        )
        
        finally:
            model.train(original_mode)
        
        return VisualizationResults(
            samples=samples,
            epoch=epoch or 0,
            debug_info=debug_info
        )
    
    def _generate_debug_info(
        self,
        sample_data: VisualizationSample,
        epoch: typing.Optional[int] = None
    ) -> typing.Dict:
        """Generate debug information for a sample."""
        primary_threshold = self.score_thresholds[0]
        predictions = sample_data.threshold_results[primary_threshold]
        num_predictions = len(predictions) if predictions else 0
        
        debug_info = {
            "num_predictions": num_predictions,
            "primary_threshold": primary_threshold,
            "motion_length": sample_data.motion_length,
            "num_prompts": len(sample_data.raw_prompts)
        }
        
        if epoch is not None:
            logger.info(f"Epoch {epoch} - Sample {sample_data.sample_index}: Generated {num_predictions} predictions with threshold {primary_threshold}")
        
        if num_predictions == 0 and primary_threshold > 0.1:
            debug_predictions = sample_data.threshold_results.get(0.1, [])
            debug_predictions_count = len(debug_predictions)
            debug_info["debug_threshold_0.1_predictions"] = debug_predictions_count
            
            if epoch is not None:
                logger.info(f"Debug: With threshold 0.1: Generated {debug_predictions_count} predictions")
        
        return debug_info
    
    def create_combined_visualization(
        self,
        sample: VisualizationSample,
        title: str,
        output_path: str
    ) -> typing.Optional[typing.Any]:
        """
        Create a combined visualization showing GT and all threshold predictions using the correct format.
        
        Args:
            sample: The sample data to visualize.
            title: Title for the visualization.
            output_path: Path to save the HTML file.
            
        Returns:
            The plotly figure object if successful, None otherwise.
        """
        try:
            # Convert ground truth spans to the correct format
            # Format: List[Tuple[str, List[Tuple[int, int, float]]]]
            groundtruth_spans = []
            for prompt_text, spans, _ in sample.raw_prompts:
                if spans:  # Only add if there are actual spans
                    # Convert spans to include score (1.0 for GT)
                    span_list_with_scores = [(start, end, 1.0) for start, end in spans]
                    groundtruth_spans.append((f"GT: {prompt_text}", span_list_with_scores))
                else:
                    # Add empty span list for prompts without spans
                    groundtruth_spans.append((f"GT: {prompt_text}", []))
            
            # Collect predictions from all thresholds in the correct format
            all_predicted_spans = []
            for threshold, threshold_predictions in sample.threshold_results.items():
                # Group predictions by prompt text
                prompt_predictions_dict = {}
                
                for prompt_text, span_list in threshold_predictions:
                    pred_prompt_key = f"PRED({threshold}): {prompt_text}"
                    if pred_prompt_key not in prompt_predictions_dict:
                        prompt_predictions_dict[pred_prompt_key] = []
                    prompt_predictions_dict[pred_prompt_key].extend(span_list)
                
                # Convert to the required format
                for pred_prompt_key, span_list in prompt_predictions_dict.items():
                    all_predicted_spans.append((pred_prompt_key, span_list))
                
                # Also add empty entries for prompts that have no predictions
                for prompt_text, _, _ in sample.raw_prompts:
                    pred_prompt_key = f"PRED({threshold}): {prompt_text}"
                    if pred_prompt_key not in prompt_predictions_dict:
                        all_predicted_spans.append((pred_prompt_key, []))
            
            # Combine all predictions for this single sample
            combined_predictions = groundtruth_spans + all_predicted_spans
            
            # Create EvaluationResult in the correct batch format (single sample)
            combined_result = EvaluationResult(
                motion_length=[sample.motion_length],
                predictions=[combined_predictions]
            )
            
            # Prepare prompt names for display (ensure all prompts are shown)
            all_gt_prompts = [f"GT: {p[0]}" for p in sample.raw_prompts]
            all_pred_prompts = []
            for threshold in self.score_thresholds:
                threshold_pred_prompts = [
                    f"PRED({threshold}): {p[0]}" 
                    for p in sample.raw_prompts
                ]
                all_pred_prompts.extend(threshold_pred_prompts)
            
            all_prompts_to_show = all_gt_prompts + all_pred_prompts
            
            if self.debug:
                logger.info(f"Creating visualization with {len(combined_predictions)} prediction groups")
                logger.info(f"Motion length: {sample.motion_length}")
                logger.info(f"All prompts to show: {len(all_prompts_to_show)}")
            
            # Create visualization - plot_evaluation_results returns a list of figures
            figures = plot_evaluation_results(
                combined_result,
                title=title,
                all_prompts=all_prompts_to_show,
            )
            
            # Get the first (and only) figure since we have a single sample
            combined_figure = figures[0] if figures and figures[0] is not None else None
            
            if combined_figure:
                combined_figure.write_html(output_path)
                if self.debug:
                    logger.info(f"Saved visualization to {os.path.basename(output_path)}")
            else:
                logger.warning(f"No figure generated for sample {sample.sample_index}")
            
            return combined_figure
            
        except Exception as exception:
            logger.error(f"Error creating visualization: {exception}")
            traceback.print_exc()
            return None