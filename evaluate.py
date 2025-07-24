import gc
import os
import tqdm
import torch
import hydra
import pytorch_lightning as pl
from hydra import main
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.constants import (
    DEFAULT_HYDRA_CONFIG_PATH,
    DEFAULT_HYDRA_VERSION_BASE
)
from src.config import read_config
from src.model.moliner import MoLiNER
from src.load import load_model_from_cfg
from src.types import RawBatch, ProcessedBatch
from src.model.metrics.iou import IntervalDetectionMetric, IOU_THRESHOLDS

@main(
    config_path=DEFAULT_HYDRA_CONFIG_PATH,
    config_name="evaluate",
    version_base=DEFAULT_HYDRA_VERSION_BASE
)
def evaluate_model(cfg: DictConfig):
    ckpt = cfg.ckpt
    device = cfg.device
    run_dir = cfg.run_dir
    score = cfg.score
    
    validation_dataset = instantiate(
        cfg.data,
        split="validation"
    )
    
    cfg = read_config(run_dir)
    model = load_model_from_cfg(
        cfg,
        ckpt_name=ckpt,
        device=device
    )
    
    validation_dataloader = instantiate(
        cfg.dataloader,
        dataset=validation_dataset,
        collate_fn=validation_dataset.collate_function,
        shuffle=False,
    )
    
    iou_metric = IntervalDetectionMetric(IOU_THRESHOLDS, score_threshold=score)
    
    model.eval()
    
    import pdb
    
    for index, raw_batch in tqdm.tqdm(enumerate(validation_dataloader), desc="[evaluation]"):
        if index == 208:
            pdb.set_trace()
        
        processed_batch = ProcessedBatch.from_raw_batch(raw_batch, model.prompts_tokens_encoder).to(device)
        
        # TODO: there is a memory leak in here
        output = model.forward(processed_batch)
        
        batch_prompts = [[prompt[0] for prompt in prompts] for prompts in raw_batch.prompts]
        evaluation_results = model.decoder.decode(
            output,
            batch_prompts,
            score
        )
        
        # Convert to the correct format for the fixed metric
        batch_predictions = []
        batch_groundtruths = []
        
        for result, prompts in zip(evaluation_results, raw_batch.prompts):
            # Format predictions: List[(prompt_text, start, end, score)]
            sample_predictions = []
            for prompt_text, start, end, score in result.predictions:
                sample_predictions.append((prompt_text, start, end, score))
            batch_predictions.append(sample_predictions)
            
            # Format ground truth: List[(prompt_text, spans_list, is_sequence_prompt)]
            sample_groundtruths = []
            for prompt_text, spans, is_sequence_prompt in prompts:
                sample_groundtruths.append((prompt_text, spans, is_sequence_prompt))
            batch_groundtruths.append(sample_groundtruths)
        
        # Update the metric with properly formatted data
        iou_metric.update(preds=batch_predictions, target=batch_groundtruths)
        
        # NOTE: very required as there is a memory leak somewhere
        del processed_batch, output, evaluation_results
        del batch_predictions, batch_groundtruths
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        
    pdb.set_trace()
    
    metrics = iou_metric.compute()
    print("[evaluation]:")
    print(metrics)

if __name__ == "__main__":
    evaluate_model()