import os
import pdb
import tqdm
import hydra
import torch
import random
import logging

import pytorch_lightning as lightning

from hydra import main
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.model import MoLiNER
from src.auth import login_to_huggingface
from src.load import load_model_from_cfg
from src.config import read_config, save_config
from src.visualizations import plot_evaluation_results

from src.constants import (
    DEFAULT_HYDRA_CONFIG_PATH,
    DEFAULT_HYDRA_VERSION_BASE,
    DEFAULT_THRESHOLD,
    DEFAULT_DEVICE
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

@main(config_path=DEFAULT_HYDRA_CONFIG_PATH, config_name="test", version_base=DEFAULT_HYDRA_VERSION_BASE)
def test(cfg: DictConfig):
    
    threshold = cfg.test.threshold if "threshold" in cfg.test else DEFAULT_THRESHOLD
    device = cfg.device if "device" in cfg else DEFAULT_DEVICE
    
    if cfg.test.pdb:
        logger.info("Running in PDB mode")
        pdb.set_trace()
    
    ckpt = None
    if cfg.run_dir is not None:
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        cfg = read_config(cfg.run_dir)
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{cfg.run_dir}")
    else:
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")

    logger.info(f"[cfg]: {cfg}")
    logger.info(f"[cfg.data]: {cfg.data}")
    logger.info(f"[cfg.model]: {cfg.model}")

    logger.info(f"[ckpt]: {ckpt}")

    lightning.seed_everything(cfg.seed)

    logger.info("[data]: loading the dataloaders")
    
    if cfg.test.data:
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
            collate_fn=train_dataset.collate_function,
            shuffle=True,
            num_workers=0
        )

        val_dataloader = instantiate(
            cfg.dataloader,
            dataset=validation_dataset,
            collate_fn=validation_dataset.collate_function,
            shuffle=False,
            num_workers=0
        )
    
        if cfg.test.preload: 
            for batch in tqdm.tqdm(iterable=train_dataloader, total=len(train_dataloader), desc="[preload-dataloader]:"):
                pass
            
            for batch in tqdm.tqdm(iterable=val_dataloader, total=len(val_dataloader), desc="[preload-dataloader]:"):
                pass
        
    if cfg.run_dir is not None:
        model = load_model_from_cfg(cfg, ckpt_name="best", device=device, eval_mode=True, pretrained=True)
        logger.info("[model]: loading the model & weights")
    else:
        model: MoLiNER = instantiate(cfg.model)
    
    logger.info("[model]: ready")
    
    from src.types import RawBatch, ProcessedBatch
    
    raw_batch = RawBatch.create_random(
        batch_size=2,
        max_frames_per_motion=200,
        max_prompts_per_motion=4,
        max_spans_per_prompt=2,
        device=torch.device(model.device)
    )

    preprocessed_batch = ProcessedBatch.from_raw_batch(
        raw_batch=raw_batch,
        encoder=model.prompts_tokens_encoder
    )
    
    outputs = model.forward(
        preprocessed_batch
    )
    
    loss = model.loss.forward(
        forward_output=outputs,
        batch=preprocessed_batch,
    )
    
    if cfg.test.pdb:
        pdb.set_trace()
    
    batch_size: int = raw_batch.transformed_motion.shape[0]

    for i in range(batch_size):
        motion_length = int(raw_batch.motion_mask[i].sum())
        motion_tensor = raw_batch.transformed_motion[i, :motion_length, :]

        prompt_texts = [prompt[0] for prompt in raw_batch.prompts[i]]

        if not prompt_texts:
            logger.warning("No prompts for this sample, skipping evaluation.")
            continue
            
        formatted_prompts = [(text, [], True) for text in prompt_texts]
        
        eval_raw_batch = RawBatch(
            sid=[0],
            dataset_name=["evaluation"],
            amass_relative_path=["none"],
            raw_motion=torch.zeros_like(motion_tensor.unsqueeze(0)),
            transformed_motion=motion_tensor.unsqueeze(0).to(device),
            motion_mask=torch.ones(1, motion_tensor.shape[0], dtype=torch.bool).to(device),
            prompts=[formatted_prompts]
        )
        
        eval_processed_batch = ProcessedBatch.from_raw_batch(
            raw_batch=eval_raw_batch,
            encoder=model.prompts_tokens_encoder
        )
        
        with torch.no_grad():
            forward_output = model.forward(
                eval_processed_batch,
                batch_index=0
            )
            print("forward_output.similarity_matrix.max()", forward_output.similarity_matrix.max())
            print("forward_output.similarity_matrix.mean()", forward_output.similarity_matrix.mean())
            print("forward_output.similarity_matrix.min()", forward_output.similarity_matrix.min())
            decoded_results = model.decoder.forward(
                forward_output=forward_output,
                prompts=prompt_texts,
                score_threshold=threshold,
            )
            evaluation_result = decoded_results[0]

        if not evaluation_result.predictions:
            logger.info("No predictions found above the score threshold.")

        figure = plot_evaluation_results(
            evaluation_result,
            title=f"Motion-Prompt Localization (Sample {i+1} - Nested Strategy)",
        )

        if figure:
            output_path = os.path.join(cfg.run_dir, f"test_timeline_sample_{i+1}.html")
            figure.write_html(output_path)
            # output_path = f"test_timeline_sample_{i+1}.png"
            # figure.write_image(output_path)
            logger.info(f"Plot saved successfully to: {os.path.abspath(output_path)}")
        else:
            logger.warning("Plot generation skipped because there were no predictions.")
            
if __name__ == "__main__":
    # NOTE: issue was that the dataset was the one moving tensors to gpu and thus CUDA was involved before dataloader
    # i modified it and set it to move to CPU and dataloader is the one responsible of moving to GPU, more specifically pytorch lightinig
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    
    test()
