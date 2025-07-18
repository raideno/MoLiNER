import torch
import typing
import typing_extensions

from ._base import LearningRateConfig, BaseOptimizer

class SGDOptimizer(BaseOptimizer):
    def __init__(self, lr: LearningRateConfig, momentum: float, weight_decay: float):
        super().__init__(lr)
        
        self.momentum = momentum
        self.weight_decay = weight_decay
    
    def configure_optimizer(self, model) -> torch.optim.Optimizer:
        param_groups = self._get_parameter_groups(model)
        
        return torch.optim.SGD(
            param_groups,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
