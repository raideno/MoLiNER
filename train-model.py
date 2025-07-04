# HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false python train-model.py

import tqdm
import hydra
import torch
import logging

import pytorch_lightning as lightning

from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config

# --- --- --- --- --- --- ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
# --- --- --- --- --- --- ---

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="train-model", version_base="1.3")
def train_model(cfg: DictConfig):
    ckpt = None
    if cfg.resume_dir is not None:
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        cfg = read_config(cfg.resume_dir)
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{cfg.resume_dir}")
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
        shuffle=True,
    )
    validation_dataloader = instantiate(
        cfg.dataloader,
        dataset=validation_dataset,
        collate_fn=validation_dataset.collate_fn,
        shuffle=False,
    )
    
    logger.info("[model]: loading the model")
    model = instantiate(cfg.model)
    
    logger.info("[model]: loading motion encoder weights")
    
    trainer = instantiate(cfg.trainer)
    
    logger.info("[training]: started")    
    
    trainer.fit(
        model,
        train_dataloader,
        validation_dataloader,
        ckpt_path=ckpt
    )

if __name__ == "__main__":
    # NOTE: issue was that the dataset was the one moving tensors to gpu and thus CUDA was involved before dataloader
    # i modified it and set it to move to CPU and dataloader is the one responsible of moving to GPU, more specifically pytorch lightinig
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    
    train_model()
