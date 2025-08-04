import abc
import torch
import typing
import typing_extensions

# NOTE: required to avoid circular dependencies issues
if typing.TYPE_CHECKING:
    from src.refactor.segmentation.segmentation import StartEndSegmentationModel

class LearningRateConfig(typing_extensions.TypedDict):
    scratch: float
    pretrained: float

class BaseOptimizer(abc.ABC):
    def __init__(self, lr: LearningRateConfig):
        self.lr = lr
    
    def _get_parameter_groups(self, model: "StartEndSegmentationModel") -> list[dict]:
        pretrained_lr = self.lr["pretrained"]
        non_pretrained_lr = self.lr["scratch"]
        
        pretrained_parameters = []
        non_pretrained_parameters = []
        
        if model.motion_encoder.pretrained:
            pretrained_parameters.extend(list(model.motion_encoder.parameters()))
        else:
            non_pretrained_parameters.extend(list(model.motion_encoder.parameters()))
        
        non_pretrained_modules = [
            model.classifier,
            model.aggregator,
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