# HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false python train-model.py model=moliner data=locate-babel

import tqdm
import hydra
import torch
import logging

import resolvers

# # --- --- --- --- --- --- ---
import os
# import sys
# sys.path.append(os.getcwd())
# # --- --- --- --- --- --- ---

import pytorch_lightning as lightning

from hydra import main
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.auth import login_to_huggingface
from src.model import MoLiNER
from src.config import read_config, save_config
from src.data.utils.collator import SimpleBatchStructureCollator

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
login_to_huggingface()
# --- --- --- --- --- --- ---

logger = logging.getLogger(__name__)

@main(config_path=DEFAULT_HYDRA_CONFIG_PATH, config_name="train-model", version_base=DEFAULT_HYDRA_VERSION_BASE)
def train_model(cfg: DictConfig):
    # NOTE: uncomment when warned about "float32 matmul precision to utilize, tensor cores efficiently"
    # torch.set_float32_matmul_precision('medium')
    # torch.set_float32_matmul_precision('high')
    
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
        
    logger.info("[model]: loading the model")
    model: MoLiNER = instantiate(cfg.model)
    
    train_dataloader: torch.utils.data.DataLoader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=SimpleBatchStructureCollator(model.prompts_tokens_encoder),
        shuffle=True,
    )
    validation_dataloader: torch.utils.data.DataLoader = instantiate(
        cfg.dataloader,
        dataset=validation_dataset,
        collate_fn=SimpleBatchStructureCollator(model.prompts_tokens_encoder),
        shuffle=False,
    )

    trainer = instantiate(cfg.trainer)
    
    # NOTE: find wandb logger instance and log model configuration
    wandb_logger = None
    for logger_instance in trainer.loggers:
        if hasattr(logger_instance, '__class__') and 'WandBLogger' in str(logger_instance.__class__):
            wandb_logger = logger_instance
            break
    
    if wandb_logger is not None:
        wandb_logger.log_model_config(cfg.model)
        
        wandb_logger.log_hyperparams(cfg)
        
        wandb_logger.watch_model(model, log_freq=100)
        
        wandb_logger.save_config_as_json(cfg, os.path.join(cfg.run_dir, "config.json"))
    
    logger.info("[model]: loading motion encoder weights")
    
    logger.info("[training]: started")    
    
    trainer.fit(
        model,
        train_dataloader,
        validation_dataloader,
        ckpt_path=ckpt
    )
    
    # NOTE: log artifacts to WandB after training
    if wandb_logger is not None:
        try:
            checkpoint_dir = os.path.join(cfg.run_dir, "checkpoints")
            if os.path.exists(checkpoint_dir):
                wandb_logger.log_artifacts(checkpoint_dir, "checkpoints")
            
        except Exception as exception:
            logger.warning(f"Failed to log artifacts to WandB: {exception}")
    
    logger.info("[training]: completed")

if __name__ == "__main__":
    # NOTE: issue was that the dataset was the one moving tensors to gpu and thus CUDA was involved before dataloader
    # i modified it and set it to move to CPU and dataloader is the one responsible of moving to GPU, more specifically pytorch lightinig
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    
    train_model()
