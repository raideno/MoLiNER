import os
import pdb
import json
import tqdm
import hydra
import torch
import gradio
import random
import logging

import pytorch_lightning as lightning

from omegaconf import DictConfig
from hydra.utils import instantiate
from src.auth import login_to_huggingface
from src.config import read_config, save_config

from src.model import MoLiNER
from src.visualizations.spans import plot_evaluation_results

from src.visualizations.spans import plot_evaluation_results

from src.data.typing import RawBatch, ProcessedBatch, EvaluationResult

# --- --- --- --- --- --- ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
# --- --- --- --- --- --- ---
login_to_huggingface()
# --- --- --- --- --- --- ---

logger = logging.getLogger(__name__)

from src.constants import (
    DEFAULT_FPS,
    DEFAULT_THRESHOLD,
)

@hydra.main(config_path="configs", config_name="interface", version_base="1.3")
def interface(cfg: DictConfig):
    train_dataset = instantiate(
        cfg.data,
        split="train"
    )
    validation_dataset = instantiate(
        cfg.data,
        split="validation"
    )
    
    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=False,
        # num_workers=4
    )
    
    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=validation_dataset,
        collate_fn=validation_dataset.collate_fn,
        shuffle=False,
        # num_workers=4
    )
    
    lightning.seed_everything(cfg.seed)
    
    logger.info("[model]: loading the model & weights")
    model: MoLiNER = instantiate(cfg.model)
    
    if cfg.ckpt is not None:
        model.load_from_checkpoint(
            cfg.model_weights,
            strict=False,
            map_location=cfg.device,
        )
    
    model = model.to(cfg.device)
    
    def gradio_plot(prompts_text: str, dataset_split: str, batch_index: int, sample_in_batch: int):
        if not prompts_text.strip():
            raise gradio.Error("Please enter at least one prompt.")
        else:
            prompts = [prompt.strip() for prompt in prompts_text.replace('\n', ',').split(',') if prompt.strip()]
            prompts = prompts[:8]
        
        if not prompts:
            raise gradio.Error("Please enter at least one valid prompt.")
        
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
        
        motion_tensor = motion_tensor.to(cfg.device)
        
        model.eval()
        
        with torch.no_grad():
            prediction_outputs = model.evaluate(
                motion=motion_tensor,
                prompts=prompts
            )
        
        
        prediction_title = f"Model Predictions - {dataset_split.title()} Batch {batch_index} Sample {sample_in_batch} ({len(prompts)} prompts)"
        prediction_figure = plot_evaluation_results(prediction_outputs, title=prediction_title)
        
        groundtruth_spans = []
        sample_prompts = raw_batch.prompts[sample_in_batch]
        
        logger.info(f"Sample prompts from batch: {sample_prompts}")
        
        if sample_prompts:
            logger.info(f"Found {len(sample_prompts)} prompt entries")
            for prompt_data in sample_prompts:
                
                if isinstance(prompt_data, (list, tuple)) and len(prompt_data) >= 2:
                    prompt_text = prompt_data[0]
                    spans = prompt_data[1] if len(prompt_data) > 1 else []
                    
                    if isinstance(spans, (list, tuple)):
                        for span in spans:
                            if isinstance(span, (list, tuple)) and len(span) >= 2:
                                start_frame, end_frame = span[0], span[1]
                                groundtruth_spans.append((prompt_text, start_frame, end_frame, 1.0))
        else:
            logger.warning("No prompts found in sample for ground truth visualization")
        
        logger.info(f"Total ground truth spans: {len(groundtruth_spans)}")
        
        if groundtruth_spans:
            groundtruth_result = EvaluationResult(motion_length=motion_length, predictions=groundtruth_spans)
            groundtruth_title = f"Ground Truth - {dataset_split.title()} Batch {batch_index} Sample {sample_in_batch} ({len(groundtruth_spans)} spans)"
            groundtruth_figure = plot_evaluation_results(groundtruth_result, title=groundtruth_title)
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
        
        return prediction_figure, groundtruth_figure, sample_json

    train_size = len(train_dataloader) if train_dataloader else 0
    val_size = len(val_dataloader) if val_dataloader else 0
    
    interface = gradio.Interface(
        fn=gradio_plot,
        inputs=[
            gradio.Textbox(
                label="Prompts (comma-separated, max 8)", 
                value="walking, running",
                placeholder="Enter prompts separated by commas, e.g., walking, running, jumping",
                lines=3
            ),
            gradio.Dropdown(
                choices=["train", "validation"],
                value="validation",
                label="Dataset Split",
                info="Select whether to use training or validation dataset"
            ),
            gradio.Number(
                label="Batch Index",
                value=0,
                precision=0,
                minimum=0,
                maximum=max(train_size-1, val_size-1) if train_size > 0 and val_size > 0 else 0,
                info=f"Training batches: {train_size}, Validation batches: {val_size}"
            ),
            gradio.Number(
                label="Sample in Batch",
                value=0,
                precision=0,
                minimum=0,
                maximum=10,
                info="Index of sample within the selected batch (usually 0 to batch_size-1)"
            )
        ],
        outputs=[
            gradio.Plot(label="Model Predictions"),
            gradio.Plot(label="Ground Truth"),
            gradio.Code(label="Sample Data (JSON)", language="json")
        ],
        title="Motion-Prompt Localization Evaluation",
        description="Select a batch and sample from the dataset to analyze. The interface shows model predictions, ground truth annotations, and sample metadata."
    )
    
    interface.launch(share=cfg.share)

if __name__ == "__main__":
    interface()