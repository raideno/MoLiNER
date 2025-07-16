# python interface.py run_dir=/home/nadir/motion-linner/out/2025-07-04_17-38-49

import os
import pdb
import json
import tqdm
import hydra
import torch
import gradio
import typing
import random
import logging

import src.auth

import pytorch_lightning as lightning

from hydra import main
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.model import MoLiNER
from src.auth import login_to_huggingface
from src.config import read_config, save_config
from src.visualizations.spans import plot_evaluation_results
from src.types import RawBatch, ProcessedBatch, EvaluationResult

from src.constants import (
    DEFAULT_HYDRA_CONFIG_PATH,
    DEFAULT_HYDRA_VERSION_BASE,
    DEFAULT_THRESHOLD
)

# --- --- --- --- --- --- ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
# --- --- --- --- --- --- ---
login_to_huggingface()
# --- --- --- --- --- --- ---

logger = logging.getLogger(__name__)

@main(config_path=DEFAULT_HYDRA_CONFIG_PATH, config_name="interface", version_base=DEFAULT_HYDRA_VERSION_BASE)
def interface(cfg: DictConfig):
    train_dataset = instantiate(
        cfg.data,
        split="train"
    )
    validation_dataset = instantiate(
        cfg.data,
        split="validation"
    )
    
    train_dataloader: torch.utils.data.DataLoader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_function,
        shuffle=False,
        # num_workers=4
    )
    
    val_dataloader: torch.utils.data.DataLoader = instantiate(
        cfg.dataloader,
        dataset=validation_dataset,
        collate_fn=validation_dataset.collate_function,
        shuffle=False,
        # num_workers=4
    )
    
    lightning.seed_everything(cfg.seed)
    
    logger.info("[model]: loading the model & weights")
    
    if cfg.run_dir is not None:
        from src.load import load_model_from_cfg
        model: MoLiNER = load_model_from_cfg(
            cfg,
            ckpt_name="best",
            device=cfg.device,
            eval_mode=True,
            pretrained=True
        )
    else:
        model: MoLiNER = instantiate(cfg.model)
        model = model.to(cfg.device)
    
    def get_sample_data(dataset_split: str, batch_index: int, sample_in_batch: int):
        """Helper function to get sample data from the dataset"""
        if dataset_split == "train":
            dataloader = train_dataloader
        else:
            dataloader = val_dataloader
        
        if batch_index < 0 or batch_index >= len(dataloader):
            raise gradio.Error(f"Batch index must be between 0 and {len(dataloader)-1} for {dataset_split} dataset.")
        
        for i, batch in enumerate(dataloader):
            if i == batch_index:
                raw_batch = batch
                break
        else:
            raise gradio.Error(f"Could not find batch {batch_index}")
        
        batch_size = raw_batch.transformed_motion.shape[0]
        if sample_in_batch < 0 or sample_in_batch >= batch_size:
            raise gradio.Error(f"Sample index must be between 0 and {batch_size-1} for this batch.")
        
        motion_length = int(raw_batch.motion_mask[sample_in_batch].sum())
        motion_tensor = raw_batch.transformed_motion[sample_in_batch, :motion_length, :]
        
        return raw_batch, motion_tensor, motion_length, batch_size

    def show_groundtruth_and_json(dataset_split: str, batch_index: int, sample_in_batch: int, source_filter: typing.List[str]):
        """Show ground truth and JSON data immediately when sample changes"""
        try:
            raw_batch, motion_tensor, motion_length, batch_size = get_sample_data(dataset_split, batch_index, sample_in_batch)
            
            groundtruth_spans = []
            groundtruth_sources = []
            sample_prompts = raw_batch.prompts[sample_in_batch]
            
            logger.info(f"Sample prompts from batch: {sample_prompts}")
            
            if sample_prompts:
                logger.info(f"Found {len(sample_prompts)} prompt entries")
                for prompt_data in sample_prompts:
                    
                    if isinstance(prompt_data, (list, tuple)) and len(prompt_data) >= 2:
                        prompt_text = prompt_data[0]
                        spans = prompt_data[1] if len(prompt_data) > 1 else []
                        
                        if len(prompt_data) >= 3:
                            is_sequence_prompt = prompt_data[2] if len(prompt_data) > 2 else False
                            source = "sequence_prompt" if is_sequence_prompt else "frame_prompt"
                        else:
                            source = "unspecified"
                        
                        if isinstance(spans, (list, tuple)):
                            for span in spans:
                                if isinstance(span, (list, tuple)) and len(span) >= 2:
                                    start_frame, end_frame = span[0], span[1]
                                    groundtruth_spans.append((prompt_text, start_frame, end_frame, 1.0))
                                    groundtruth_sources.append(source)
            else:
                logger.warning("No prompts found in sample for ground truth visualization")
            
            logger.info(f"Total ground truth spans: {len(groundtruth_spans)}")
            
            if groundtruth_spans:
                groundtruth_result = EvaluationResult(motion_length=motion_length, predictions=groundtruth_spans)
                groundtruth_title = f"Ground Truth - {dataset_split.title()} Batch {batch_index} Sample {sample_in_batch} ({len(groundtruth_spans)} spans)"
                groundtruth_figure = plot_evaluation_results(
                    groundtruth_result, 
                    title=groundtruth_title,
                    sources=groundtruth_sources,
                    filter_sources=source_filter if source_filter else None
                )
            else:
                raise gradio.Error(f"No ground truth annotations found for sample {sample_in_batch} in batch {batch_index}.")
            
            sample_info = {
                "batch_index": batch_index,
                "sample_in_batch": sample_in_batch,
                "motion_length": motion_length,
                "batch_size": batch_size,
                "sample_id": raw_batch.sid[sample_in_batch] if hasattr(raw_batch, 'sid') else "unknown",
                "prompts": sample_prompts,
                "motion_shape": list(motion_tensor.shape)
            }
            sample_json = json.dumps(sample_info, indent=2, default=str)
            
            return groundtruth_figure, sample_json
            
        except Exception as e:
            logger.error(f"Error in show_groundtruth_and_json: {str(e)}")
            return None, f"Error: {str(e)}"

    cached_forward_data = {"output": None, "prompts": None, "sample_key": None}
    
    def get_forward_output(motion_tensor: torch.Tensor, prompts: typing.List[str], sample_key: str):
        """Get or compute forward output, using cache when possible"""
        if (cached_forward_data["output"] is not None and 
            cached_forward_data["prompts"] == prompts and 
            cached_forward_data["sample_key"] == sample_key):
            return cached_forward_data["output"]
        
        model.eval()
        
        formatted_prompts = [(text, [], True) for text in prompts]
        
        raw_batch = RawBatch(
            sid=[0],
            dataset_name=["evaluation"],
            amass_relative_path=["none"],
            # NOTE: dummy raw motion as we don't need it for evaluation
            raw_motion=torch.zeros_like(motion_tensor.unsqueeze(0)),
            transformed_motion=motion_tensor.unsqueeze(0).to(cfg.device),
            motion_mask=torch.ones(1, motion_tensor.shape[0], dtype=torch.bool).to(cfg.device),
            prompts=[formatted_prompts]
        )
        
        processed_batch = ProcessedBatch.from_raw_batch(
            raw_batch=raw_batch,
            encoder=model.prompts_tokens_encoder
        )
        
        with torch.no_grad():
            forward_output = model.forward(
                processed_batch,
                batch_index=0
            )
        
        cached_forward_data["output"] = forward_output
        cached_forward_data["prompts"] = prompts.copy()
        cached_forward_data["sample_key"] = sample_key
        
        return forward_output

    def clear_cache():
        """Clear the forward output cache"""
        cached_forward_data["output"] = None
        cached_forward_data["prompts"] = None
        cached_forward_data["sample_key"] = None

    def show_prediction(prompts_text: str, threshold: float, dataset_split: str, batch_index: int, sample_in_batch: int):
        """Show model predictions when button is clicked"""
        if not prompts_text.strip():
            raise gradio.Error("Please enter at least one prompt.")
        else:
            prompts = [prompt.strip() for prompt in prompts_text.replace('\n', ',').split(',') if prompt.strip()]
            prompts = prompts[:8]
        
        if not prompts:
            raise gradio.Error("Please enter at least one valid prompt.")
        
        try:
            raw_batch, motion_tensor, motion_length, batch_size = get_sample_data(dataset_split, batch_index, sample_in_batch)
            
            motion_tensor = motion_tensor.to(cfg.device)
            
            sample_key = f"{dataset_split}_{batch_index}_{sample_in_batch}"
            
            forward_output = get_forward_output(motion_tensor, prompts, sample_key)
            
            with torch.no_grad():
                decoded_results = model.decoder.decode(
                    forward_output=forward_output,
                    prompts=prompts,
                    score_threshold=threshold,
                )
                prediction_outputs = decoded_results[0]
            
            prediction_title = f"Model Predictions - {dataset_split.title()} Batch {batch_index} Sample {sample_in_batch} ({len(prompts)} prompts, threshold={threshold:.3f})"
            prediction_figure = plot_evaluation_results(prediction_outputs, title=prediction_title)
            
            return prediction_figure
            
        except Exception as e:
            logger.error(f"Error in show_prediction: {str(e)}")
            raise gradio.Error(f"Error generating prediction: {str(e)}")

    train_size = len(train_dataloader) if train_dataloader else 0
    val_size = len(val_dataloader) if val_dataloader else 0
    
    with gradio.Blocks(title="Motion-Prompt Localization Evaluation") as interface:
        gradio.Markdown("# Motion-Prompt Localization Evaluation")
        gradio.Markdown("Select a batch and sample from the dataset to analyze. The interface shows model predictions, ground truth annotations, and sample metadata.")
        
        with gradio.Accordion("Model Configuration", open=False):
            from omegaconf import OmegaConf
            model_config_dict = OmegaConf.to_container(cfg.model, resolve=True)
            model_config_json = json.dumps(model_config_dict, indent=2, ensure_ascii=False, sort_keys=True)
            gradio.Code(value=model_config_json, label="Model Configuration", language="json")
        
        with gradio.Row():
            with gradio.Column():
                prompts_input = gradio.Textbox(
                    label="Prompts (comma-separated, max 8)", 
                    value="walking, running",
                    placeholder="Enter prompts separated by commas, e.g., walking, running, jumping",
                    lines=3
                )
                
                threshold_slider = gradio.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=DEFAULT_THRESHOLD,
                    step=0.01,
                    label="Score Threshold",
                    info="Confidence threshold for predictions (0.0 to 1.0)"
                )
                
                source_filter = gradio.CheckboxGroup(
                    choices=["unspecified", "raw_label", "sequence_prompt", "frame_prompt", "proc_label", "act_cat"],
                    value=["unspecified", "raw_label", "sequence_prompt", "frame_prompt", "proc_label", "act_cat"],
                    label="Source Filter (Ground Truth)",
                    info="Select which sources to display in ground truth visualization"
                )
                
                dataset_split_input = gradio.Dropdown(
                    choices=["train", "validation"],
                    value="validation",
                    label="Dataset Split",
                    info="Select whether to use training or validation dataset"
                )
                
                batch_index_input = gradio.Number(
                    label="Batch Index",
                    value=0,
                    precision=0,
                    minimum=0,
                    maximum=max(train_size-1, val_size-1) if train_size > 0 and val_size > 0 else 0,
                    info=f"Training batches: {train_size}, Validation batches: {val_size}"
                )
                
                sample_in_batch_input = gradio.Number(
                    label="Sample in Batch",
                    value=0,
                    precision=0,
                    minimum=0,
                    maximum=10,
                    info="Index of sample within the selected batch (usually 0 to batch_size-1)"
                )
                
                predict_button = gradio.Button("Generate Predictions", variant="primary")
        
        with gradio.Row():
            with gradio.Column():
                prediction_plot = gradio.Plot(label="Model Predictions")
            
            with gradio.Column():
                groundtruth_plot = gradio.Plot(label="Ground Truth")
                
        sample_json_output = gradio.Code(label="Sample Data (JSON)", language="json")
        
        # NOTE: update ground truth and JSON whenever sample parameters change
        def on_sample_change(*args):
            clear_cache()
            return show_groundtruth_and_json(*args)
            
        for input_component in [dataset_split_input, batch_index_input, sample_in_batch_input, source_filter]:
            input_component.change(
                fn=on_sample_change,
                inputs=[dataset_split_input, batch_index_input, sample_in_batch_input, source_filter],
                outputs=[groundtruth_plot, sample_json_output]
            )
        
        # NOTE: clear cache when prompts change
        def on_prompts_change(*args):
            clear_cache()
            
        prompts_input.change(fn=on_prompts_change)
        
        # NOTE: generate predictions when button is clicked OR when threshold changes
        predict_button.click(
            fn=show_prediction,
            inputs=[prompts_input, threshold_slider, dataset_split_input, batch_index_input, sample_in_batch_input],
            outputs=[prediction_plot]
        )
        
        # NOTE: Auto-update predictions when threshold changes (if we already have prompts)
        threshold_slider.change(
            fn=show_prediction,
            inputs=[prompts_input, threshold_slider, dataset_split_input, batch_index_input, sample_in_batch_input],
            outputs=[prediction_plot]
        )
        
        interface.load(
            fn=show_groundtruth_and_json,
            inputs=[dataset_split_input, batch_index_input, sample_in_batch_input, source_filter],
            outputs=[groundtruth_plot, sample_json_output]
        )
    
    interface.launch(share=cfg.share)

if __name__ == "__main__":
    interface()