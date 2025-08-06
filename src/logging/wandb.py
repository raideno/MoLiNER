import os
import json
import pytorch_lightning
import pytorch_lightning.loggers

class WandbLogger(pytorch_lightning.loggers.WandbLogger):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        with open(os.path.join(kwargs['save_dir'], 'config.json'), 'r') as file:
            config = json.load(file)
            
        model_name: str = kwargs['project']
            
        if model_name == 'MoLiNER':
            motion_frames_encoder = config["model"]["motion_frames_encoder"]
            prompts_tokens_encoder = config["model"]["prompts_tokens_encoder"]
            tags = [
                f"{motion_frames_encoder['_target_'].split('.')[-1]}(pretrained={motion_frames_encoder.get('pretrained', False)}, frozen={motion_frames_encoder.get('frozen', False)})",
                f"{prompts_tokens_encoder['_target_'].split('.')[-1]}(pretrained={prompts_tokens_encoder.get('pretrained', False)}, frozen={prompts_tokens_encoder.get('frozen', False)})",
                config["model"]["spans_generator"]["_target_"].split(".")[-1],
                config["model"]["prompt_representation_layer"]["_target_"].split(".")[-1],
                config["model"]["span_representation_layer"]["_target_"].split(".")[-1],
                config["model"]["scorer"]["_target_"].split(".")[-1],
                config["model"]["decoder"]["_target_"].split(".")[-1],
                config["model"]["loss"]["_target_"].split(".")[-1],
                config["model"]["optimizer"]["_target_"].split(".")[-1],
                # config["model"]["postprocessors"]["_target_"].split(".")[-1],
            ]
        elif model_name == 'StartEndSegmentationModel':
            motion_encoder = config['model']['motion_encoder']
            tags = [
                f"{motion_encoder['_target_'].split('.')[-1]}(pretrained={motion_encoder.get('pretrained', False)}, frozen={motion_encoder.get('frozen', False)})",
                config["model"]["classifier"]["_target_"].split(".")[-1],
                config["model"]["aggregator"]["_target_"].split(".")[-1],
                config["model"]["optimizer"]["_target_"].split(".")[-1],
                config["model"]["loss"]["_target_"].split(".")[-1],
                f"window-size={config['model']['window_size']}",
                f"window-size{config['model']['window_size']}",
            ]
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        print("[tags]:", tags)
            
        super().__init__(
            *args,
            **kwargs,
            config=config,
            tags=tags
        )
    