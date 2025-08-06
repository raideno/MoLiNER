import abc
import pytorch_lightning

from src.types import (
    Batch,
    EvaluationResult,
)

class BaseModel(pytorch_lightning.LightningModule, abc.ABC):
    @abc.abstractmethod
    def predict(
        self,
        batch: Batch,
        threshold: float
    )-> EvaluationResult:
        pass