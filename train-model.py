# NOTE: contain custom hydra resolvers

import hydra
import torch
import typing
import logging
import omegaconf
import pytorch_lightning
import pytorch_lightning.loggers

import src.resolvers

from src.auth import login_to_huggingface
from src.config import read_config, save_config
from src.models import MoLiNER, StartEndSegmentationModel
from src.data.utils.collator import SimpleBatchStructureCollator

from src.constants import (
    DEFAULT_HYDRA_CONFIG_PATH,
    DEFAULT_HYDRA_VERSION_BASE
)

# --- --- --- --- --- --- ---
login_to_huggingface()
# --- --- --- --- --- --- ---

logger = logging.getLogger(__name__)

# type: ignore
@hydra.main(config_path=DEFAULT_HYDRA_CONFIG_PATH, config_name="train-model", version_base=DEFAULT_HYDRA_VERSION_BASE)
def train_model(cfg: omegaconf.DictConfig):
    logger.debug(f"[cfg]: {cfg}")
    
    logger.info(f"[run_dir]: {cfg.run_dir}")
    
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

    logger.info(f"[ckpt]: {ckpt}")

    pytorch_lightning.seed_everything(cfg.seed)

    logger.info("[data]: loading the dataloaders")
    
    train_dataset = hydra.utils.instantiate(
        cfg.data,
        split="train"
    )
    validation_dataset = hydra.utils.instantiate(
        cfg.data,
        split="validation"
    )
        
    logger.info("[model]: loading the model")
    model: MoLiNER | StartEndSegmentationModel = hydra.utils.instantiate(cfg.model)
    
    train_dataloader: torch.utils.data.DataLoader = hydra.utils.instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=SimpleBatchStructureCollator(model.prompts_tokens_encoder if isinstance(model, MoLiNER) else None),
        shuffle=True,
    )
    validation_dataloader: torch.utils.data.DataLoader = hydra.utils.instantiate(
        cfg.dataloader,
        dataset=validation_dataset,
        collate_fn=SimpleBatchStructureCollator(model.prompts_tokens_encoder if isinstance(model, MoLiNER) else None),
        shuffle=False,
    )

    trainer = hydra.utils.instantiate(cfg.trainer)
    
    logger.info("[model]: loading motion encoder weights")
    
    logger.info("[training]: started")    
    
    trainer.fit(
        model,
        train_dataloader,
        validation_dataloader,
        ckpt_path=ckpt
    )
    
    logger.info("[training]: completed")

if __name__ == "__main__":
    train_model()
