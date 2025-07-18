import typing

from ._base import BasePipeline

from src.data.utils.filtering import (
    FilterConfig,
    FilterFunction
)
from src.data.utils.batching import babel_simplify_batch_structure

class BabelPipeline(BasePipeline):
    def __init__(self, name: typing.Optional[str] = None):
        super().__init__(name or "babel")
        
        self.add_step(babel_simplify_batch_structure, batched=False)
        
class __BabelFromSourcePipeline(BabelPipeline):
    def __init__(self, source: str, name: typing.Optional[str] = None):
        super().__init__(name or f"babel-{source}")
        
        # NOTE: we consider only spans from the specified source
        filter_function = FilterFunction(FilterConfig(
            min_prompts_per_sample=1,
            sources=[source],
        ))
        
        self.add_step(filter_function, batched=True)

class BabelActionCategoryPipeline(__BabelFromSourcePipeline):
    def __init__(self):
        super().__init__(source="act_cat")
        
class BabelProcLabelPipeline(__BabelFromSourcePipeline):
    def __init__(self):
        super().__init__(source="proc_label")
        
class BabelRawLabelPipeline(__BabelFromSourcePipeline):
    def __init__(self):
        super().__init__(source="raw_label")