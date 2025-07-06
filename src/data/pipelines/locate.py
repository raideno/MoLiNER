from src.data.pipelines.index import BabelPipeline

from src.data.utils.filtering import (
    create_filter_function,
    create_locate_classes_filter_function,
    FilterConfig
)

class LocatePipeline(BabelPipeline):
    """
    Pipeline for processing Babel dataset with LOCATE filtering.
    Applies simplification and then filters for LOCATE classes.
    """
    
    def __init__(self):
        super().__init__("locate")
        
        locate_filter_config = FilterConfig(
            min_motion_frames=8,
            max_motion_frames=4096,
            min_prompts_per_sample=1,
            max_prompts_per_sample=4,
            split_max_prompts_per_sample=True,
            prompt_text_filter_function=create_locate_classes_filter_function(),
            min_span_frames=1,
            max_span_frames=32,
            max_spans_per_prompt=8,
            sources=["act_cat"],
            debug=False
        )

        locate_filter_fn = create_filter_function(locate_filter_config)
        self.add_step(locate_filter_fn, batched=True)