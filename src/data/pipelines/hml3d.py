import typing

from ._base import BasePipeline

from src.data.utils.filtering import FilterConfig, FilterFunction
from src.data.utils.batching import hml3d_simplify_batch_structure

class HML3DPipeline(BasePipeline):
    def __init__(self, name: typing.Optional[str] = None):
        super().__init__(name or "hml3d")
        
        self.add_step(hml3d_simplify_batch_structure, batched=True)
        
class Max1024HML3DPipeline_(HML3DPipeline):
    def __init__(self, splitted: bool, name: str):
        super().__init__(name)
        
        self.splitted = splitted
                
        filter_function = FilterFunction(FilterConfig(
            # seed=None
            # fps=None
            min_motion_frames=1,
            max_motion_frames=1024,
            min_prompts_per_sample=1,
            # max_prompts_per_sample=1,
            # split_max_prompts_per_sample=True,
            # prompt_text_filter_function=None,
            # min_span_frames=1,
            # max_span_frames=64,
            # min_spans_per_prompt=None
            # max_spans_per_prompt=None
            sources=["texts"],
            annotation_types=["sequence"]
            # debug: bool = False
        ))
        
        if splitted:
            filter_function.config.max_prompts_per_sample = 1
            filter_function.config.split_max_prompts_per_sample = True
        
        self.add_step(filter_function, batched=True)
        
class Max1024HML3DSplittedPipeline(Max1024HML3DPipeline_):
    def __init__(self):
        super().__init__(splitted=True, name="max-1024-hml3d-splitted")
        
class Max1024HML3DGroupedPipeline(Max1024HML3DPipeline_):
    def __init__(self):
        super().__init__(splitted=False, name="max-1024-hml3d-grouped")
