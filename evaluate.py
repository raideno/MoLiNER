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
    # test_dataset = instantiate(
    #     cfg.data,
    #     split="test"
    # )
    
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
    # test_dataloader: torch.utils.data.DataLoader = instantiate(
    #     cfg.dataloader,
    #     dataset=test_dataset,
    #     collate_fn=test_dataset.collate_function,
    #     shuffle=True,
    # )

    iou_metric = IntervalDetectionMetric(IOU_THRESHOLDS, score_threshold=score)
    
    all_groundtruths, all_predictions = [], []
    for raw_batch in tqdm.tqdm(validation_dataloader, desc="[evaluation]:"):
        processed_batch = ProcessedBatch.from_raw_batch(raw_batch, model.prompts_tokens_encoder).to(device)
        output = model.forward(processed_batch)

        batch_prompts = [[prompt[0] for prompt in prompts] for prompts in raw_batch.prompts]

        evaluation_results = model.decoder.decode(
            output,
            batch_prompts,
            score
        )
        
        batch_predictions = []
        batch_groundtruths = []
        
        for result, prompts in zip(evaluation_results, raw_batch.prompts):
            groundtruth_spans = [span for (_, spans, _) in prompts for span in spans]
            predicted_spans = [(start, end) for (_, start, end, _) in result.predictions]
            
            batch_groundtruths.append(groundtruth_spans)
            batch_predictions.append(predicted_spans)
        
        iou_metric.update(preds=batch_predictions, target=batch_groundtruths)

    metrics = iou_metric.compute()
    
    print("[evaluation]:")
    print(metrics)

if __name__ == "__main__":
    evaluate_model()