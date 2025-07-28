import torch
import torch.nn as nn
import typing

from ._base import BaseClassifier

class MLPClassifier(BaseClassifier):
    def __init__(
        self,
        latent_dim=256,
        hidden_dim=128
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        
        self.start_regression_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        
        self.end_regression_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        encoded_features: torch.Tensor,
        motion_masks: typing.Optional[torch.Tensor] = None,
        batch_index: typing.Optional[int] = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE: encoded_features is of shape (B, LatentDimension)
        B, latent_dim = encoded_features.size()
        
        class_logits = self.classification_head(encoded_features)
        start_logits = self.start_regression_head(encoded_features)
        end_logits = self.end_regression_head(encoded_features)
        
        return class_logits, start_logits, end_logits