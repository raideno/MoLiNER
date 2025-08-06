from omegaconf import OmegaConf

OmegaConf.register_new_resolver("split-get-last", lambda s: s.split('.')[-1])