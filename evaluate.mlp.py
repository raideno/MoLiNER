import pdb
# HYDRA_FULL_ERROR=1 python evaluate.py -m \
#   device=cuda:0 score=0.5 \
#   protocol=self,locate,mlp \
#   run_dir="./out/training.2025-07-26_11-39-16","./out/training.2025-07-26_14-27-48","./out/training.2025-07-27_00-19-11","./out/training.2025-07-27_02-43-23","./out/training.2025-07-27_11-48-31","./out/training.2025-07-27_20-51-52"

from src.types import EvaluationResult
import itertools
import gc
import os
import tqdm
import json
import torch
import hydra
import pprint
import typing
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

from mlp_helpers import TALLEvaluator

@main(
    config_path=DEFAULT_HYDRA_CONFIG_PATH,
    config_name="evaluate.mlp",
    version_base=DEFAULT_HYDRA_VERSION_BASE
)
def evaluate_model(cfg: DictConfig):
    ckpt = cfg.ckpt
    device = cfg.device
    run_dir = cfg.run_dir
    score = cfg.score
    # protocol = cfg.protocol
    
    cfg = read_config(run_dir)

    # NOTE: use MLP data; HumanML3D sequences
    from src.data.hml3d import HML3DDataset
    from src.helpers.motion_normalizer import MotionNormalizer
    validation_dataset = HML3DDataset(
        split="test",
        pipeline="mlp-max-1024-hml3d-splitted",
        motion_normalizer=MotionNormalizer(stats_path="./statistics/hml3d.pt"),
    )
   
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
    # --- --- ---
    model.postprocessors = []
    from src.model.modules.decoders.greedy import GreedyDecoder, DecodingStrategy
    model.decoder = GreedyDecoder(strategy=DecodingStrategy.FLAT)
    # --- --- ---
    
    model.eval()
    
    from mlp_helpers import TALLEvaluator
    
    evaluator = TALLEvaluator()
    
    results: typing.List[EvaluationResult] = []
    groundtruths = []
    
    for index, raw_batch in tqdm.tqdm(enumerate(validation_dataloader), total=len(validation_dataloader), desc="[evaluation]"):
        raw_batch: RawBatch = raw_batch.to(device)
        
        evaluation_results = model.predict(
            raw_batch=raw_batch,
            threshold=score
        )
        
        for element in raw_batch.prompts:
            if element and element[0][1]:  # check both the prompt list and spans list
                span = element[0][1][0]  # first span from the first prompt
            else:
                span = []
            groundtruths.append(span)
        
        results.append(evaluation_results)
        
        # del processed_batch, output, evaluation_results
        del evaluation_results
        torch.cuda.empty_cache()
        gc.collect()
    
    validation_dataloader: typing.List[RawBatch] = validation_dataloader
    
    prompt_index = 0
    span_index = 0
    
    # NOTE: a list of batches
    predictions = [
        [
            element[prompt_index][1][span_index][:2] if len(element) > 0 else (0.0, 0.0)
            for element in result.predictions
        ] for result in results
    ]
    # groundtruths = [
    #     [
    #         element[prompt_index][1][span_index][:2] if len(element) > 0 else []
    #         for element in batch.prompts
    #     ] for batch in validation_dataloader
    # ]
    
    predictions_ = [item for sublist in predictions for item in sublist]
    # groundtruths_ = [item for sublist in groundtruths for item in sublist]
    groundtruths_ = groundtruths
    
    _predictions_ = [[prediction] for prediction in predictions_]
    _groundtruths_ = [[groundtruth] for groundtruth in groundtruths_]
    
    all_rank1, all_rank5, r1_m_iou, r5_m_iou = evaluator.eval(
        predictions=_predictions_,
        groundtruths=_groundtruths_    
    )
        
    print("[all_rank1]:")
    print(all_rank1)

    print("[all_rank5]:")
    print(all_rank5)

    print("[r1_m_iou]:")
    print(r1_m_iou)

    print("[r5_m_iou]:")
    print(r5_m_iou)

    nb = len(_groundtruths_)

    for k, v in all_rank1.items():
        print(f"[{k}]:", v / nb * 100, 1)
    for k, v in all_rank5.items():
        print(f"[{k}]:", v / nb * 100, 1)


if __name__ == "__main__":
    evaluate_model()
