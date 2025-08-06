import gc
import os
import json
import tqdm
import hydra
import torch
import hydra
import typing
import pprint
import datetime
import omegaconf

from src.constants import (
    DEFAULT_HYDRA_CONFIG_PATH,
    DEFAULT_HYDRA_VERSION_BASE
)

from src.config import read_config
from src.load import load_model_from_cfg
from src.types import Batch, EvaluationResult
from src.models import MoLiNER, StartEndSegmentationModel
from src.data.utils.collator import SimpleBatchStructureCollator
from src.metrics.locate import evaluate_interval_detection, IOU_THRESHOLDS

# type: ignore
@hydra.main(
    config_path=DEFAULT_HYDRA_CONFIG_PATH,
    config_name="evaluate.locate",
    version_base=DEFAULT_HYDRA_VERSION_BASE
)
def evaluate_locate(cfg: omegaconf.DictConfig):
    ckpt = cfg.ckpt
    device = cfg.device
    run_dir = cfg.run_dir
    score = cfg.score
    
    cfg = read_config(run_dir)
   
    model: typing.Union[MoLiNER, StartEndSegmentationModel] = load_model_from_cfg(
        cfg,
        ckpt_name=ckpt,
        device=device
    )
    if not isinstance(model, MoLiNER) and not isinstance(model, StartEndSegmentationModel):
        raise ValueError("The model must be an instance of MoLiNER or StartEndSegmentationModel for LOCATE evaluation.")
    
    from src.data.babel import BabelDataset
    from src.helpers.motion_normalizer import MotionNormalizer
    validation_dataset = BabelDataset(
        split="validation",
        pipeline="locate",
        motion_normalizer=MotionNormalizer(stats_path="./statistics/babel.pt"),
    )
    validation_dataloader = hydra.utils.instantiate(
        cfg.dataloader,
        dataset=validation_dataset,
        collate_fn=SimpleBatchStructureCollator(model.prompts_tokens_encoder if isinstance(model, MoLiNER) else None),
        shuffle=False,
    )
    
    model.eval()
    
    all_evaluation_results: typing.List[EvaluationResult] = []
    all_ground_truth_batches: typing.List[typing.List[typing.List[typing.Tuple[str, typing.List[typing.Tuple[int, int]], bool]]]] = []
    
    for index, batch in tqdm.tqdm(enumerate(validation_dataloader), total=len(validation_dataloader), desc="[evaluation]"):
        batch = batch.to(device)
        
        evaluation_results = model.predict(
            batch=batch,
            threshold=score
        )
        
        all_evaluation_results.append(evaluation_results)
        
        # NOTE: store the ground truth batch (convert to the expected format)
        batch_ground_truth = []
        batch_size = len(batch.prompts)
        
        for batch_idx in range(batch_size):
            sample_targets = batch.prompts[batch_idx]
            batch_ground_truth.append(sample_targets)
        
        all_ground_truth_batches.append(batch_ground_truth)
        
        del evaluation_results
        torch.cuda.empty_cache()
        gc.collect()
    
    metrics = evaluate_interval_detection(
        evaluation_results=all_evaluation_results,
        ground_truth_batches=all_ground_truth_batches,
    )
    
    print("[evaluation]:")
    pprint.pprint(metrics)
    
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint": ckpt,
        "device": device,
        "score_threshold": score,
        "metrics": metrics,
    }
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(run_dir, f"evaluation_{score}_{timestamp_str}.json")
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=2, default=str)
    print(f"[evaluation]: results saved to {results_path}")

if __name__ == "__main__":
    evaluate_locate()
