import typing

from ._base import BasePipeline

from src.data.utils.batching import hml3d_simplify_batch_structure

class HML3DPipeline(BasePipeline):
    def __init__(self):
        super().__init__("hml3d")
        
        self.add_step(hml3d_simplify_batch_structure, batched=True)