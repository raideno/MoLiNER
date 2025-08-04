import pdb
import torch
import typing
import logging
import torchmetrics
import pytorch_lightning

from src.constants import LOCATE_CLASSES_DICT
from src.helpers.segmentation_to_evaluation_result import segmentation_predictions_to_evaluation_result
from src.types import (
    RawBatch,
    SegmenterForwardOutput,
    EvaluationResult,
)
from src.models.modules import (
    BaseOptimizer
)

from .modules import (
    BaseMotionEncoder,
    BaseClassifier,
    BaseLoss,
    BaseAggregator,
)
from .helpers import (
    create_windows,
)
from .modules.losses.standard import extract_window_labels

logger = logging.getLogger(__name__)

class StartEndSegmentationModel(pytorch_lightning.LightningModule):
    def __init__(
        self,
        motion_encoder: BaseMotionEncoder,
        classifier: BaseClassifier,
        aggregator: BaseAggregator,
        optimizer: BaseOptimizer,
        loss: BaseLoss,
        window_size: int,
        **kwargs,
    ):
        super().__init__()

        self.motion_encoder: BaseMotionEncoder = motion_encoder
        self.classifier: BaseClassifier = classifier
        self.aggregator: BaseAggregator = aggregator
        
        self.optimizer: BaseOptimizer = optimizer
        self.window_size: int = window_size
        
        self.loss: BaseLoss = loss
        
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=20)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=20)

    def configure_optimizers(self):
        return self.optimizer.configure_optimizer(self)

    def forward(
        self,
        *args,
        **kwargs
    ) -> SegmenterForwardOutput:
        batch: RawBatch = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
                
        windowed_motion, window_metadata, windows_per_sample = create_windows(self.window_size, 1, batch)

        total_windows = windowed_motion.shape[0]

        latent_features = self.motion_encoder.forward(
            motion_features=windowed_motion,
            motion_masks=torch.ones(total_windows, self.window_size, device=windowed_motion.device),
            batch_index=batch_index
        )
        
        class_logits, start_logits, end_logits = self.classifier.forward(
            encoded_features=latent_features,
            motion_masks=torch.ones(total_windows, self.window_size, device=windowed_motion.device),
            batch_index=batch_index
        )
        
        start_logits = start_logits.view(total_windows, -1)
        end_logits = end_logits.view(total_windows, -1)
        
        return SegmenterForwardOutput(
            class_logits=class_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            windows_positions=window_metadata,
            windows_per_sample=windows_per_sample,
            batch_size=batch.transformed_motion.shape[0]
        )
    
    def training_step(self, *args, **kwargs):
        batch: "RawBatch" = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        batch_size = batch.motion_mask.size(0)
        
        output = self.forward(batch, batch_index=batch_index)
        
        loss = self.loss.forward(output, batch, batch_index)
        
        class_probabilities = torch.softmax(output.class_logits, dim=-1)
        class_predictions = torch.argmax(class_probabilities, dim=-1)
        
        with torch.no_grad():
            labels = extract_window_labels(batch, output)
            gt_labels = labels[:, 0].long()
            
            valid_mask = gt_labels != -1
            if torch.any(valid_mask):
                self.train_accuracy(class_predictions[valid_mask], gt_labels[valid_mask])
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        
        return {
            "loss": loss
        }
        
    def validation_step(self, *args, **kwargs):
        batch: "RawBatch" = args[0]
        batch_index: int = kwargs.get("batch_index", 0)
        
        batch_size = batch.motion_mask.size(0)
        
        output = self.forward(batch, batch_index=batch_index)
        
        loss = self.loss.forward(output, batch, batch_index)
        
        class_probabilities = torch.softmax(output.class_logits, dim=-1)
        class_predictions = torch.argmax(class_probabilities, dim=-1)
        
        with torch.no_grad():
            labels = extract_window_labels(batch, output)
            gt_labels = labels[:, 0].long()
            
            valid_mask = gt_labels != -1
            if torch.any(valid_mask):
                self.val_accuracy(class_predictions[valid_mask], gt_labels[valid_mask])
        
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log("val/accuracy", self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size, sync_dist=True)
        
        return {
            "loss": loss
        }
        
    def predict(
        self,
        batch: RawBatch,
        threshold: float
    ) -> EvaluationResult:
        self.eval()
        
        with torch.no_grad():
            output = self.forward(batch)
            
            # NOTE: (total_windows, num_classes)
            class_probs = torch.softmax(output.class_logits, dim=-1)
            
            predictions = []
            
            window_idx = 0
            for batch_idx in range(output.batch_size):
                motion_length = batch.motion_mask[batch_idx].sum().item()
                num_windows = output.windows_per_sample[batch_idx].item()
                
                motion_windows = output.windows_positions[window_idx:window_idx + num_windows]
                motion_class_probs = class_probs[window_idx:window_idx + num_windows]
                
                frame_predictions = self.aggregator.forward(
                    motion_length=motion_length,
                    window_metadata=motion_windows,
                    class_probs=motion_class_probs,
                    threshold=threshold
                )
                
                predictions.append(frame_predictions)
                window_idx += num_windows
            
            return segmentation_predictions_to_evaluation_result(
                predictions=predictions,
                class_names=[",".join(value) for value in LOCATE_CLASSES_DICT.values()]
            )
