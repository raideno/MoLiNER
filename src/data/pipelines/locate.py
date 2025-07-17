import math

from .babel import BabelPipeline

from src.data.utils.filtering import (
    FilterConfig,
    FilterFunction,
    create_locate_classes_filter_function,
)

VERY_BIG_INT = int(1e9)

class LocatePipeline(BabelPipeline):
    """
    Pipeline for processing Babel dataset with LOCATE filtering.
    Applies simplification and then filters for LOCATE classes.
    """
    
    def __init__(self):
        super().__init__("locate")
        
        # NOTE: only filtering is that we take act_cat only and we limit motion frames between 8 and 4096
        locate_filter_config = FilterConfig(
            min_motion_frames=8,
            max_motion_frames=4096,
            min_prompts_per_sample=1,
            max_prompts_per_sample=VERY_BIG_INT,
            split_max_prompts_per_sample=True,
            prompt_text_filter_function=create_locate_classes_filter_function(),
            min_span_frames=1,
            max_span_frames=VERY_BIG_INT,
            max_spans_per_prompt=VERY_BIG_INT,
            sources=["act_cat"],
            debug=False
        )

        locate_filter_function = FilterFunction(locate_filter_config)
        self.add_step(locate_filter_function, batched=True)