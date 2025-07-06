import tqdm
import hydra
import torch
import logging

import pytorch_lightning as lightning

from hydra import main
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.config import read_config, save_config

from src.constants import (
    DEFAULT_HYDRA_CONFIG_PATH,
    DEFAULT_HYDRA_VERSION_BASE
)

# --- --- --- --- --- --- ---
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warnings, 3=errors
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')
# --- --- --- --- --- --- ---

logger = logging.getLogger(__name__)

@main(config_path=DEFAULT_HYDRA_CONFIG_PATH, config_name="test-model", version_base=DEFAULT_HYDRA_VERSION_BASE)
def test_model(cfg: DictConfig):
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

    logger.info(f"[cfg.run_dir]: {cfg.run_dir}")
    logger.info(f"[cfg.data]: {cfg.data}")
    logger.info(f"[cfg.model.classifier]: {cfg.model.classifier}")
    logger.info(f"[cfg.model.motion_encoder]: {cfg.model.motion_encoder}")

    logger.info(f"[ckpt]: {ckpt}")

    lightning.seed_everything(cfg.seed)

    logger.info("[data]: loading the dataloaders")
    
    train_dataset = instantiate(
        cfg.data,
        split="train"
    )
    val_dataset = instantiate(
        cfg.data,
        split="val"
    )
    
    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_function,
        shuffle=True,
    )
    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        collate_fn=val_dataset.collate_function,
        shuffle=False,
    )
    
    for batch in tqdm.tqdm(iterable=train_dataloader, total=len(train_dataloader), desc="[preload-dataloader]:"):
        pass
    
    for batch in tqdm.tqdm(iterable=val_dataloader, total=len(val_dataloader), desc="[preload-dataloader]:"):
        pass
    
    logger.info("[model]: loading the model")
    model = instantiate(cfg.model)
    
    logger.info("[model]: loading motion encoder weights")
    
    trainer = instantiate(cfg.trainer)
    
    logger.info("[training]: started")    
    
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        ckpt_path=ckpt
    )

if __name__ == "__main__":
    # NOTE: issue was that the dataset was the one moving tensors to gpu and thus CUDA was involved before dataloader
    # i modified it and set it to move to CPU and dataloader is the one responsible of moving to GPU, more specifically pytorch lightinig
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    
    test_model()
