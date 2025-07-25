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
    
    for index, raw_batch in tqdm.tqdm(enumerate(validation_dataloader), total=len(validation_dataloader), desc="[evaluation]"):
        raw_batch = raw_batch.to(device)
        
        evaluation_results = model.predict(
            raw_batch=raw_batch,
            threshold=score
        )
        
        batch_predictions = []
        batch_targets = []
        
        batch_size = len(raw_batch.prompts)
        
        for batch_idx in range(batch_size):
            sample_targets = raw_batch.prompts[batch_idx]
            
            sample_predictions = []
            motion_predictions = (
                evaluation_results.predictions[batch_idx] 
                if batch_idx < len(evaluation_results.predictions) 
                else []
            )
            
            for prompt_text, span_list in motion_predictions:
                for start_frame, end_frame, score_val in span_list:
                    sample_predictions.append((prompt_text, start_frame, end_frame, score_val))
            
            batch_predictions.append(sample_predictions)
            batch_targets.append(sample_targets)
        
        iou_metric.update(preds=batch_predictions, target=batch_targets)
        
        # del processed_batch, output, evaluation_results
        del evaluation_results
        del batch_predictions, batch_targets
        torch.cuda.empty_cache()
        gc.collect()
    
    pdb.set_trace()
    
    metrics = iou_metric.compute()
    
    print("[evaluation]:")
    print(metrics)

if __name__ == "__main__":
    evaluate_model()
