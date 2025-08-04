import abc
import torch
import typing
import typing_extensions

class LearningRateConfig(typing_extensions.TypedDict):
    scratch: float
    pretrained: float

class BaseOptimizer(abc.ABC):
    def __init__(self, lr: LearningRateConfig):
        self.lr = lr
    
    def _get_parameter_groups(self, model) -> list[dict]:
        pretrained_lr = self.lr["pretrained"]
        non_pretrained_lr = self.lr["scratch"]
        
        pretrained_parameters = []
        non_pretrained_parameters = []
        
        if model.prompts_tokens_encoder.pretrained:
            pretrained_parameters.extend(list(model.prompts_tokens_encoder.parameters()))
        else:
            non_pretrained_parameters.extend(list(model.prompts_tokens_encoder.parameters()))
            
        if model.motion_frames_encoder.pretrained:
            pretrained_parameters.extend(list(model.motion_frames_encoder.parameters()))
        else:
            non_pretrained_parameters.extend(list(model.motion_frames_encoder.parameters()))
        
        non_pretrained_modules = [
            model.spans_generator,
            model.prompt_representation_layer,
            model.span_representation_layer,
            model.scorer,
            model.decoder,
            model.loss
        ]
        
        for module in non_pretrained_modules:
            non_pretrained_parameters.extend(list(module.parameters()))
        
        param_groups = []
        
        if len(pretrained_parameters) > 0:
            param_groups.append({
                'params': pretrained_parameters,
                'lr': pretrained_lr,
                'name': 'pretrained'
            })
        
        if len(non_pretrained_parameters) > 0:
            param_groups.append({
                'params': non_pretrained_parameters,
                'lr': non_pretrained_lr,
                'name': 'non_pretrained'
            })
        
        return param_groups
    
    @abc.abstractmethod
    def configure_optimizer(self, model) -> torch.optim.Optimizer:
        pass