from src.data.pipelines.index import BabelPipeline

from src.data.utils.filtering import (
    create_filter_function,
    create_locate_classes_filter_function,
    FilterConfig
)
from src.data.utils.augmentation import (
    standardize_spans_sliding_window,
    standardize_spans_chunking
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

        sliding_window_fn = standardize_spans_sliding_window(
            target_span_length=SPAN_LENGTH,
            max_extend_frames=MAX_EXTEND_FRAMES,
            debug=DEBUG
        )
        self.add_step(sliding_window_fn, batched=False)
    
        locate_filter_fn = create_filter_function(FILTER_CONFIG)
        self.add_step(locate_filter_fn, batched=True)
        
class ChunkingStandardizedLocatePipeline(BabelPipeline):
    """
    Pipeline for processing Babel dataset with LOCATE filtering.
    Applies simplification and then filters for LOCATE classes.
    """
    
    def __init__(self):
        super().__init__("locate")

        chunking_fn = standardize_spans_chunking(
            target_span_length=SPAN_LENGTH,
            max_extend_frames=MAX_EXTEND_FRAMES,
            debug=DEBUG
        )
        self.add_step(chunking_fn, batched=False)
    
        locate_filter_fn = create_filter_function(FILTER_CONFIG)
        self.add_step(locate_filter_fn, batched=True)