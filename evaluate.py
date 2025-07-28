# HYDRA_FULL_ERROR=1 python evaluate.py -m \
#     device=cuda:0 score=0.5 \
#     data=babel/20/base\
#     run_dir="./out/training.2025-07-26_11-39-16","./out/training.2025-07-26_14-27-48","./out/training.2025-07-27_00-19-11","./out/training.2025-07-27_02-43-23","./out/training.2025-07-27_11-48-31","./out/training.2025-07-27_20-51-52"

# HYDRA_FULL_ERROR=1 python evaluate.py -m \
#     device=cuda:0 score=0.5 \
#     run_dir="./out/training.2025-07-26_11-39-16","./out/training.2025-07-26_14-27-48","./out/training.2025-07-27_00-19-11","./out/training.2025-07-27_02-43-23","./out/training.2025-07-27_11-48-31","./out/training.2025-07-27_20-51-52"

import gc
import os
import tqdm
import torch
import hydra
import pprint
import datetime

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

    # NOTE: we use the specified data if provided, otherwise we use the data used while training the model
    data = cfg.data if "data" in cfg else None
    data = data if data is not None and data != "none" else None
    cfg = read_config(run_dir)
    data = cfg.data if data is None else data

    validation_dataset = instantiate(
        cfg.data,
        split="validation"
    )
    
    validation_dataloader = instantiate(
        cfg.dataloader,
        dataset=validation_dataset,
        collate_fn=validation_dataset.collate_function,
        shuffle=False,
    )
    
    model = load_model_from_cfg(
        cfg,
        ckpt_name=ckpt,
        device=device
    )
    
    iou_metric = IntervalDetectionMetric(IOU_THRESHOLDS, score_threshold=score)
    model.eval()
    
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
    
    metrics = iou_metric.compute()
    
    print("[evaluation]:")
    pprint.pprint(metrics)
    
    pipeline_name = data.pipeline if "pipeline" in cfg.data else "unknown"
    
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint": ckpt,
        "device": device,
        "pipeline": pipeline_name,
        "score_threshold": score,
        "metrics": metrics,
        "config": {
            "data": data,
            "run_dir": run_dir
        }
    }
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(run_dir, results_filename, f"evaluation_{pipeline_name}_{timestamp_str}.json")

    with open(results_path, 'w') as file:
        json.dump(results, file, indent=2, default=str)

    print(f"[evaluation]: results saved to {results_path}")


if __name__ == "__main__":
    evaluate_model()
