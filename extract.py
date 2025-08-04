import hydra
import logging
import omegaconf

from src.load import extract_best_ckpt, extract_ckpt
from src.constants import (
    DEFAULT_HYDRA_CONFIG_PATH,
    DEFAULT_HYDRA_VERSION_BASE
)

logger = logging.getLogger(__name__)

# type: ignore
@hydra.main(config_path=DEFAULT_HYDRA_CONFIG_PATH, config_name="extract", version_base=DEFAULT_HYDRA_VERSION_BASE)
def extract(cfg: omegaconf.DictConfig):
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
