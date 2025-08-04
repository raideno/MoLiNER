import abc
import pytorch_lightning

from src.types import (
    RawBatch,
    EvaluationResult,
)

class BaseModel(pytorch_lightning.LightningModule, abc.ABC):
    @abc.abstractmethod
    def predict(
        self,
        batch: RawBatch,
        threshold: float
    )-> EvaluationResult:
        pass