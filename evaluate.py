# HYDRA_FULL_ERROR=1 python evaluate.py -m \
#   device=cuda:0 score=0.5 \
#   protocol=self,locate,mlp \
#   run_dir="./out/training.2025-07-26_11-39-16","./out/training.2025-07-26_14-27-48","./out/training.2025-07-27_00-19-11","./out/training.2025-07-27_02-43-23","./out/training.2025-07-27_11-48-31","./out/training.2025-07-27_20-51-52"

import json
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
from src.types import RawBatch
from src.config import read_config
from src.model.moliner import MoLiNER
from src.load import load_model_from_cfg
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
    protocol = cfg.protocol
    
    cfg = read_config(run_dir)
    
    # NOTE: use the same data used during training
    if protocol == "self":
        validation_dataset = instantiate(
            cfg.data,
            split="validation"
        )
    # NOTE: use locate data
    elif protocol == "locate":
        from src.data.babel import BabelDataset
        from src.helpers.motion_normalizer import MotionNormalizer
        validation_dataset = BabelDataset(
            split="validation",
            pipeline="locate",
            motion_normalizer=MotionNormalizer(stats_path="./statistics/babel.pt"),
        )
    # NOTE: use MLP data; HumanML3D sequences
    elif protocol == "mlp":
        from src.data.hml3d import HML3DDataset
        from src.helpers.motion_normalizer import MotionNormalizer
        validation_dataset = HML3DDataset(
            split="test",
            pipeline="max-1024-hml3d-splitted",
            motion_normalizer=MotionNormalizer(stats_path="./statistics/hml3d.pt"),
        )
    else:
        raise ValueError(f"Unknown protocol: {protocol}. Valid protocols are: self, locate, mlp.")
        
    validation_dataloader = instantiate(
        cfg.dataloader,
        dataset=validation_dataset,
        collate_fn=validation_dataset.collate_function,
        shuffle=False,
    )
    
    model: MoLiNER = load_model_from_cfg(
        cfg,
        ckpt_name=ckpt,
        device=device
    )
    model.postprocessors = []
    from src.model.modules.decoders.greedy import GreedyDecoder, DecodingStrategy
    model.decoder = GreedyDecoder(strategy=DecodingStrategy.FLAT)
    
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
    
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint": ckpt,
        "device": device,
        "protocol": protocol,
        "score_threshold": score,
        "metrics": metrics,
    }
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(run_dir, f"evaluation_{protocol}_{score}_{timestamp_str}.json")

    with open(results_path, 'w') as file:
        json.dump(results, file, indent=2, default=str)

    print(f"[evaluation]: results saved to {results_path}")


if __name__ == "__main__":
    evaluate_model()
