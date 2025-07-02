# HYDRA_FULL_ERROR=1 python extract.py --multirun \
#       run_dir=

import logging
import hydra

from omegaconf import DictConfig

from src.load import extract_best_ckpt, extract_ckpt

logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="extract", version_base="1.3")
def extract(cfg: DictConfig):
    run_dir = cfg.run_dir
    ckpt = cfg.ckpt
    mode = cfg.mode

    logger.info(f"[extracter]: mode={mode}")
    
    logger.info("[extracter]: extracting the checkpoint...")
    
    if mode == "best":
        extract_best_ckpt(run_dir)
    elif mode == "default":
        extract_ckpt(run_dir, ckpt_name=ckpt)
    else:
        raise Exception(f"Unsupported mode: {mode}. Use 'best' or 'default'.")
    
    logger.info("[extractor]: done extracting the checkpoint.")

if __name__ == "__main__":
    extract()
