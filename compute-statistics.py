import os
import tqdm
import torch
import importlib

import numpy as np

from hydra import main
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.constants import (
    DEFAULT_HYDRA_CONFIG_PATH,
    DEFAULT_HYDRA_VERSION_BASE,
)

@main(config_path=DEFAULT_HYDRA_CONFIG_PATH, config_name="compute-statistics", version_base=DEFAULT_HYDRA_VERSION_BASE)
def main(cfg: DictConfig):
    dataset = instantiate(
        cfg.data,
        split="train",
        motion_normalizer=None
    )

    all_motion = {"new_joint_vecs": [], "new_joints": []}
    
    for item in tqdm.tqdm(dataset, desc="Collecting motion data"):
        motion = item.get("motion", None)
        if motion is None:
            continue
        if isinstance(motion, dict):
            if "new_joint_vecs" in motion:
                all_motion["new_joint_vecs"].append(np.array(motion["new_joint_vecs"]))
            if "new_joints" in motion:
                all_motion["new_joints"].append(np.array(motion["new_joints"]))
        else:
            all_motion["new_joint_vecs"].append(np.array(motion))
            
    stats = {"mean": {}, "std": {}}
    
    for k, arrs in all_motion.items():
        if arrs:
            concat = np.concatenate([a.reshape(-1, a.shape[-1]) if a.ndim > 2 else a for a in arrs], axis=0)
            stats["mean"][k] = concat.mean(axis=0)
            stats["std"][k] = concat.std(axis=0) + 1e-8
            
    torch.save(stats, cfg.output_path)
    
    print(f"Saved stats to {os.path.abspath(cfg.output_path)}")

if __name__ == "__main__":
    main()
