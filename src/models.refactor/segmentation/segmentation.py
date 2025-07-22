import gc
import pdb
import torch
import logging
import torch.nn as nn

from typing import Dict
from torch import Tensor

from pytorch_lightning import LightningModule

from .modules.classifier import BaseClassifier
from .modules.encoder import BaseMotionEncoder 

logger = logging.getLogger(__name__)

class StartEndSegmentationModel(LightningModule):
    def __init__(
        self,
        motion_encoder: BaseMotionEncoder,
        classifier: BaseClassifier,
        
        optimizer: BaseOpitmizer,
        
        loss: BaseLoss
    ):
        super().__init__()

        self.motion_encoder: BaseMotionEncoder = motion_encoder
        self.classifier: BaseClassifier = classifier
        
        self.optimizer: BaseOptimizer = optimizer
        
        self.loss: BaseLoss = loss
        
        self.classification_loss_fn = nn.BCEWithLogitsLoss()
        self.start_regression_loss_fn = nn.MSELoss()
        self.end_regression_loss_fn = nn.MSELoss()

    def configure_optimizers(self):
        return self.optimizer.configure_optimizer(self)

    def forward(self, batch, batch_index) -> Tensor:
        # preprocessed_motion, motion, transition_mask = batch
        
        preprocessed_motion = batch["transformed_motion"]
        motion = batch["motion"]
        transition_mask = batch["transition_mask"]
        
        latent = self.motion_encoder(batch)
        class_logits, start_logits, end_logits = self.classifier(latent)
        
        return class_logits, start_logits, end_logits
    
    def compute_loss(self, class_logits, start_logits, end_logits, label) -> Tensor:
        # pdb.set_trace()
        
        class_loss = self.classification_loss_fn(class_logits, label[:, 0])

        valid_start_mask = label[:, 1] != -1
        valid_end_mask = label[:, 2] != -1

        if valid_start_mask.any():
            start_loss = self.start_regression_loss_fn(start_logits[valid_start_mask], label[:, 1][valid_start_mask])
        else:
            start_loss = torch.tensor(0.0, device=label.device)

        if valid_end_mask.any():
            end_loss = self.end_regression_loss_fn(end_logits[valid_end_mask], label[:, 2][valid_end_mask])
        else:
            end_loss = torch.tensor(0.0, device=label.device)

        loss = class_loss + start_loss + end_loss

        return loss, (class_loss, start_loss, end_loss)

    def step(self, batch, batch_idx) -> Tensor:
        # NOTE: preprocessed_motion (B, window_size, 263)
        # NOTE: motion (B, WINDOW_SIZE, 22, 3)
        # NOTE: transition_mask (B,)
        # preprocessed_motion, motion, transition_mask = batch
        
        preprocessed_motion = batch["transformed_motion"]
        motion = batch["motion"]
        transition_mask = batch["transition_mask"]
        
        # pdb.set_trace()
        
        label = self.extract_transition_info(transition_mask)
        
        class_logits, start_logits, end_logits = self(batch, batch_idx)
        
        class_logits = class_logits.view(-1)
        start_logits = start_logits.view(-1)
        end_logits = end_logits.view(-1)
    
        loss, (class_loss, start_loss, end_loss) = self.compute_loss(class_logits, start_logits, end_logits, label)
        
        return loss, (class_loss, start_loss, end_loss)
    
    def training_step(self, batch, batch_idx: int) -> Tensor:
        # logger.info(f"Before forward: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        loss, (class_loss, start_loss, end_loss) = self.step(batch, batch_idx)
        # logger.info(f"After forward / step: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_class_loss", class_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_start_loss", start_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_end_loss", end_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        loss, (class_loss, start_loss, end_loss) = self.step(batch, batch_idx)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_class_loss", class_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_start_loss", start_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_end_loss", end_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def extract_transition_info(self, transition_masks: Tensor) -> Tensor:
        """
        Given a batch of transition masks (B, T), where each mask is a 1D tensor of 0s and 1s indicating transitions,
        returns a tensor of shape (B, 3) for each batch element:
        [has_transition (0 or 1), start_transition (0-1 or -1), end_transition (0-1 or -1)]
        """
        B, T = transition_masks.shape
        result = torch.full((B, 3), -1.0, device=transition_masks.device)

        for i in range(B):
            mask = transition_masks[i]
            indices = torch.nonzero(mask, as_tuple=False).view(-1)
            if indices.numel() > 0:
                start = indices[0].item() / (T - 1)
                end = indices[-1].item() / (T - 1)
                result[i, 0] = 1.0
                result[i, 1] = start
                result[i, 2] = end
            else:
                result[i, 0] = 0.0

        return result