import typing

from ._base import BasePipeline

from src.data.utils.batching import babel_simplify_batch_structure

class BabelPipeline(BasePipeline):
    def __init__(self, name: typing.Optional[str] = None):
        super().__init__(name or "babel")
        
        self.add_step(babel_simplify_batch_structure, batched=False)