import typing

from ._base import BasePipeline

from src.data.utils.filtering import FilterConfig, FilterFunction
from src.data.utils.batching import hml3d_simplify_batch_structure

class HML3DPipeline(BasePipeline):
    def __init__(self, name: typing.Optional[str] = None):
        super().__init__(name or "hml3d")
        
        self.add_step(hml3d_simplify_batch_structure, batched=True)
        
class Max1024HML3DPipeline(HML3DPipeline):
    """
    Pipeline for processing HumanML3D dataset with short length filtering.
    """
    
    def __init__(self):
        super().__init__("max-1024-hml3d")
                
        filter_function = FilterFunction(FilterConfig(
            # seed: typing.Optional[int] = DEFAULT_SEED
            # fps: typing.Optional[int] = DEFAULT_FPS
            min_motion_frames=1,
            max_motion_frames=1024,
            min_prompts_per_sample=1,
            max_prompts_per_sample=1,
            split_max_prompts_per_sample=True,
            # prompt_text_filter_function,
            # min_span_frames=1,
            # max_span_frames=64,
            # min_spans_per_prompt: typing.Optional[int] = None
            # max_spans_per_prompt: typing.Optional[int] = None
            sources=["texts"],
            annotation_types=["sequence"]
            # debug: bool = False
        ))
        
        self.add_step(filter_function, batched=True)