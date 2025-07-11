from .babel import BabelPipeline

from src.data.utils.filtering import (
    FilterConfig,
    FilterFunction,
    create_locate_classes_filter_function,
)
from src.data.utils.augmentation import (
    StandardizeSpansSlidingWindow,
    StandardizeSpansChunking
)

SPAN_LENGTH = 16
MAX_EXTEND_FRAMES = 4
DEBUG = False
FILTER_CONFIG = FilterConfig(
    min_motion_frames=16,
    max_motion_frames=4096,
    min_prompts_per_sample=1,
    max_prompts_per_sample=16,
    split_max_prompts_per_sample=False,
    prompt_text_filter_function=create_locate_classes_filter_function(),
    min_span_frames=1,
    max_span_frames=SPAN_LENGTH,
    max_spans_per_prompt=16,
    sources=["act_cat"],
    debug=False
)

class WindowingStandardizedLocatePipeline(BabelPipeline):
    """
    Pipeline for processing Babel dataset with LOCATE filtering.
    Applies simplification and then filters for LOCATE classes.
    """
    
    def __init__(self):
        super().__init__("locate")

        sliding_window_function = StandardizeSpansSlidingWindow(
            target_span_length=SPAN_LENGTH,
            max_extend_frames=MAX_EXTEND_FRAMES,
            debug=DEBUG
        )
        self.add_step(sliding_window_function, batched=False)
    
        locate_filter_function = FilterFunction(FILTER_CONFIG)
        self.add_step(locate_filter_function, batched=True)
        
class ChunkingStandardizedLocatePipeline(BabelPipeline):
    """
    Pipeline for processing Babel dataset with LOCATE filtering.
    Applies simplification and then filters for LOCATE classes.
    """
    
    def __init__(self):
        super().__init__("locate")

        chunking_function = StandardizeSpansSlidingWindow(
            target_span_length=SPAN_LENGTH,
            max_extend_frames=MAX_EXTEND_FRAMES,
            debug=DEBUG
        )
        self.add_step(chunking_function, batched=False)
    
        locate_filter_function = FilterFunction(FILTER_CONFIG)
        self.add_step(locate_filter_function, batched=True)