import copy

from .babel import BabelPipeline

from src.data.utils.filtering import (
    FilterConfig,
    FilterFunction,
    create_locate_classes_filter_function,
)

from src.data.utils.augmentation import (
    StandardizeSpansSlidingWindow,
    SeparateFrameAndSequenceSpans,
)

class FilteredLocatePipeline(BabelPipeline):
    def __init__(self):
        super().__init__("locate")
        
        filter_function = FilterFunction(FilterConfig(
            # seed: typing.Optional[int] = DEFAULT_SEED
            # fps: typing.Optional[int] = DEFAULT_FPS
            min_motion_frames=1,
            max_motion_frames=1024,
            min_prompts_per_sample=1,
            max_prompts_per_sample=16,
            # split_max_prompts_per_sample: bool = False
            prompt_text_filter_function=create_locate_classes_filter_function(),
            min_span_frames=1,
            max_span_frames=64,
            # min_spans_per_prompt: typing.Optional[int] = None
            # max_spans_per_prompt: typing.Optional[int] = None
            sources=["act_cat"],
            annotation_types=["frames"]
            # debug: bool = False
        ))
        
        self.add_step(filter_function, batched=True)
        
SPAN_LENGTH = 16
MAX_EXTEND_FRAMES = 4
# NOTE: only filtering is that we take act_cat only and we limit motion frames between 8 and 4096
FILTER_CONFIG = FilterConfig(
    min_motion_frames=1,
    max_motion_frames=1024,
    min_prompts_per_sample=1,
    prompt_text_filter_function=create_locate_classes_filter_function(),
    min_span_frames=1,
    # max_span_frames,
    sources=["act_cat"],
    annotation_types=["frames", "sequence"]
)

class LocatePipeline(BabelPipeline):
    """
    Pipeline for processing Babel dataset with LOCATE filtering.
    """
    
    def __init__(self):
        super().__init__("locate")

        filter_config = copy.deepcopy(FILTER_CONFIG)
        
        locate_filter_function = FilterFunction(filter_config)
        self.add_step(locate_filter_function, batched=True)
        self.add_step(SeparateFrameAndSequenceSpans(), batched=True)
        
class WindowingStandardizedLocatePipeline(BabelPipeline):
    """
    Pipeline for processing Babel dataset with LOCATE filtering and make all spans have the same length using a sliding window.
    """
    
    def __init__(self):
        super().__init__("windowed-locate")

        sliding_window_function = StandardizeSpansSlidingWindow(
            target_span_length=SPAN_LENGTH,
            max_extend_frames=MAX_EXTEND_FRAMES,
        )
        self.add_step(sliding_window_function, batched=False)
    
        filter_config = copy.deepcopy(FILTER_CONFIG)
        filter_config.max_span_frames = SPAN_LENGTH
        
        locate_filter_function = FilterFunction(filter_config)
        self.add_step(locate_filter_function, batched=True)
        
class ChunkingStandardizedLocatePipeline(BabelPipeline):
    """
    Pipeline for processing Babel dataset with LOCATE filtering and make all spans have the same length by chunking them.
    """
    
    def __init__(self):
        super().__init__("chunked-locate")

        chunking_function = StandardizeSpansSlidingWindow(
            target_span_length=SPAN_LENGTH,
            max_extend_frames=MAX_EXTEND_FRAMES,
        )
        self.add_step(chunking_function, batched=False)
        
        filter_config = copy.deepcopy(FILTER_CONFIG)
        filter_config.max_span_frames = SPAN_LENGTH
    
        locate_filter_function = FilterFunction(FILTER_CONFIG)
        self.add_step(locate_filter_function, batched=True)