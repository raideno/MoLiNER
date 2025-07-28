import typing

from ._base import BasePipeline

from src.data.utils.filtering import (
    FilterConfig,
    FilterFunction,
    NoTransitionFilter,
)
from src.data.utils.augmentation import (
    SeparateFrameAndSequenceSpans
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
            min_motion_frames=1,
            min_prompts_per_sample=1,
            sources=[source],
            min_span_frames=1,
            annotation_types=["frames", "sequence"]
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
        
class BabelSeparate(BabelPipeline):
    def __init__(self):
        super().__init__("custom-babel")
        filter_function = FilterFunction(FilterConfig(
            min_motion_frames=1,
            max_motion_frames=1024,
            min_prompts_per_sample=1,
            max_prompts_per_sample=16,
            # split_max_prompts_per_sample=False
            prompt_text_filter_function=NoTransitionFilter(),
            min_span_frames=1,
            max_span_frames=256,
            # min_spans_per_prompt: typing.Optional[int] = None
            # max_spans_per_prompt: typing.Optional[int] = None
            sources=["raw_label"],
            annotation_types=["frames", "sequence"]
        ))
        
        
        fn = SeparateFrameAndSequenceSpans()
        # NOTE: we split the frame annotation and sequence annotations into distinct samples to not confuse the model
        self.add_step(fn, batched=True)
        
        # NOTE: we filter the samples based on the specified configuration
        self.add_step(filter_function, batched=True)