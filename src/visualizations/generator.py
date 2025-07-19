import os
import torch
import logging
import typing
from dataclasses import dataclass

from src.model import MoLiNER
from src.types import RawBatch, ProcessedBatch, EvaluationResult
from src.visualizations.spans import plot_evaluation_results

logger = logging.getLogger(__name__)


@dataclass
class VisualizationSample:
    sample_index: int
    threshold_results: typing.Dict[float, EvaluationResult]
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
        Generate visualization data for a batch of samples.
        
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
                for i in range(len(raw_batch.sid)):
                    sample_data = self._process_single_sample(
                        model=model,
                        raw_batch=raw_batch,
                        sample_index=i,
                        epoch=epoch
                    )
                    
                    if sample_data is not None:
                        samples[i] = sample_data
                        
                        if self.debug and debug_info is not None:
                            debug_info[i] = self._generate_debug_info(
                                model=model,
                                sample_data=sample_data,
                                raw_batch=raw_batch,
                                sample_index=i,
                                epoch=epoch
                            )
        
        finally:
            model.train(original_mode)
        
        return VisualizationResults(
            samples=samples,
            epoch=epoch or 0,
            debug_info=debug_info
        )
    
    def _process_single_sample(
        self,
        model: MoLiNER,
        raw_batch: RawBatch,
        sample_index: int,
        epoch: typing.Optional[int] = None
    ) -> typing.Optional[VisualizationSample]:
        motion_length = int(raw_batch.motion_mask[sample_index].sum())
        motion_tensor = raw_batch.transformed_motion[sample_index, :motion_length, :].to(model.device)
        prompt_texts = [prompt[0] for prompt in raw_batch.prompts[sample_index]]
        
        if not prompt_texts:
            logger.warning(f"Sample {sample_index} has no prompts. Skipping.")
            return None
        
        if self.debug and epoch is not None:
            logger.info(f"Epoch {epoch} - Sample {sample_index}: Motion tensor shape: {motion_tensor.shape}")
            logger.info(f"Epoch {epoch} - Sample {sample_index}: Prompts: {prompt_texts[:3]}...")
        
        formatted_prompts = [(text, [], True) for text in prompt_texts]
        eval_raw_batch = RawBatch(
            sid=[0],
            dataset_name=["evaluation"],
            amass_relative_path=["none"],
            raw_motion=torch.zeros_like(motion_tensor.unsqueeze(0)),
            transformed_motion=motion_tensor.unsqueeze(0).to(model.device),
            motion_mask=torch.ones(1, motion_tensor.shape[0], dtype=torch.bool).to(model.device),
            prompts=[formatted_prompts]
        )
        
        eval_processed_batch = ProcessedBatch.from_raw_batch(
            raw_batch=eval_raw_batch,
            encoder=model.prompts_tokens_encoder
        )
        
        forward_output = model.forward(eval_processed_batch, batch_index=0)
        
        threshold_results = {}
        for threshold in self.score_thresholds:
            decoded_results = model.decoder.decode(
                forward_output=forward_output,
                prompts=prompt_texts,
                score_threshold=threshold,
            )
            threshold_results[threshold] = decoded_results[0]
        
        return VisualizationSample(
            sample_index=sample_index,
            threshold_results=threshold_results,
            motion_length=motion_length,
            raw_prompts=raw_batch.prompts[sample_index]
        )
    
    def _generate_debug_info(
        self,
        model: MoLiNER,
        sample_data: VisualizationSample,
        raw_batch: RawBatch,
        sample_index: int,
        epoch: typing.Optional[int] = None
    ) -> typing.Dict:
        """Generate debug information for a sample."""
        primary_threshold = self.score_thresholds[0]
        evaluation_result = sample_data.threshold_results[primary_threshold]
        num_predictions = len(evaluation_result.predictions) if evaluation_result.predictions else 0
        
        debug_info = {
            "num_predictions": num_predictions,
            "primary_threshold": primary_threshold,
            "motion_length": sample_data.motion_length,
            "num_prompts": len([prompt[0] for prompt in raw_batch.prompts[sample_index]])
        }
        
        if epoch is not None:
            logger.info(f"Epoch {epoch} - Sample {sample_index}: Generated {num_predictions} predictions with threshold {primary_threshold}")
        
        if num_predictions == 0 and primary_threshold > 0.1:
            prompt_texts = [prompt[0] for prompt in raw_batch.prompts[sample_index]]
            
            motion_length = int(raw_batch.motion_mask[sample_index].sum())
            motion_tensor = raw_batch.transformed_motion[sample_index, :motion_length, :].to(model.device)
            formatted_prompts = [(text, [], True) for text in prompt_texts]
            
            eval_raw_batch = RawBatch(
                sid=[0],
                dataset_name=["evaluation"],
                amass_relative_path=["none"],
                raw_motion=torch.zeros_like(motion_tensor.unsqueeze(0)),
                transformed_motion=motion_tensor.unsqueeze(0).to(model.device),
                motion_mask=torch.ones(1, motion_tensor.shape[0], dtype=torch.bool).to(model.device),
                prompts=[formatted_prompts]
            )
            
            eval_processed_batch = ProcessedBatch.from_raw_batch(
                raw_batch=eval_raw_batch,
                encoder=model.prompts_tokens_encoder
            )
            
            forward_output = model.forward(eval_processed_batch, batch_index=0)
            
            debug_decoded_results = model.decoder.decode(
                forward_output=forward_output,
                prompts=prompt_texts,
                score_threshold=0.1,
            )
            debug_result = debug_decoded_results[0]
            debug_predictions = len(debug_result.predictions) if debug_result.predictions else 0
            
            debug_info["debug_threshold_0.1_predictions"] = debug_predictions
            
            if epoch is not None:
                logger.info(f"Debug: With threshold 0.1: Generated {debug_predictions} predictions")
        
        return debug_info
    
    def create_combined_visualization(
        self,
        sample: VisualizationSample,
        title: str,
        output_path: str
    ) -> typing.Optional[typing.Any]:
        """
        Create a combined visualization showing GT and all threshold predictions.
        
        Args:
            sample: The sample data to visualize.
            title: Title for the visualization.
            output_path: Path to save the HTML file.
            
        Returns:
            The plotly figure object if successful, None otherwise.
        """
        try:
            groundtruth_spans = [
                (f"GT: {p[0]}", s[0], s[1], 1.0) 
                for p in sample.raw_prompts 
                for s in p[1]
            ]
            
            all_predicted_spans = []
            all_pred_prompts = []
            
            for threshold, threshold_evaluation_result in sample.threshold_results.items():
                # NOTE: refix predicted spans with "PRED(<threshold>): "
                predicted_spans = [
                    (f"PRED({threshold}): {pred[0]}", pred[1], pred[2], pred[3]) 
                    for pred in threshold_evaluation_result.predictions
                ]
                all_predicted_spans.extend(predicted_spans)
                
                threshold_pred_prompts = [
                    f"PRED({threshold}): {p[0]}" 
                    for p in sample.raw_prompts
                ]
                all_pred_prompts.extend(threshold_pred_prompts)
            
            combined_predictions = groundtruth_spans + all_predicted_spans
            combined_result = EvaluationResult(
                motion_length=sample.motion_length, 
                predictions=combined_predictions
            )
            
            all_gt_prompts = [f"GT: {p[0]}" for p in sample.raw_prompts]
            all_prompts_to_show = all_gt_prompts + all_pred_prompts
            
            combined_figure = plot_evaluation_results(
                combined_result,
                title=title,
                all_prompts=all_prompts_to_show,
            )
            
            if combined_figure:
                combined_figure.write_html(output_path)
                if self.debug:
                    logger.info(f"Saved visualization to {os.path.basename(output_path)}")
            
            return combined_figure
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
