import os
import json

from omegaconf import DictConfig, OmegaConf, ListConfig

def save_config(cfg: DictConfig) -> str:
    path = os.path.join(cfg.run_dir, "config.json")
    config = OmegaConf.to_container(cfg, resolve=True)
    
    with open(path, "w") as file:
        string = json.dumps(config, indent=4)
        file.write(string)
    
    return path

def read_config(run_dir: str, return_json=False) -> DictConfig:
    path = os.path.join(run_dir, "config.json")
    
    with open(path, "r") as f:
        config = json.load(f)
    
    if return_json:
        return config
    
    config = OmegaConf.create(config)
    config.run_dir = run_dir
    
    assert isinstance(config, DictConfig), "Config should be a DictConfig instance"
    
    return config
