import torch
import typing
import typing_extensions

from ._base import LearningRateConfig, BaseOptimizer

class AdamOptimizer(BaseOptimizer):
    def configure_optimizer(self, model) -> torch.optim.Optimizer:
        param_groups = self._get_parameter_groups(model)
        
        return torch.optim.AdamW(param_groups)
