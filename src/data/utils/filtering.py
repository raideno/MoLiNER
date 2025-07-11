import pdb
import random
import typing
import logging
import dataclasses

from src.constants import DEFAULT_FPS, DEFAULT_SEED

from src.constants import (
    BABEL_20_CLASSES,
    BABEL_60_CLASSES,
    BABEL_90_CLASSES,
    BABEL_120_CLASSES,
    LOCATE_CLASSES,
)

logger = logging.getLogger(__name__)

class NoTransitionFilter:
    """
    Callable class that keeps a prompt if it does NOT contain any transition keywords.
    This is picklable and safe for multiprocessing.
    """
    def __init__(self):
        self.transition_keywords = {"transition"}

    def __call__(self, text: str) -> bool:
        """
        Returns True if the text does not contain transition keywords.
        """
        return not any(keyword in text for keyword in self.transition_keywords)

class ExactMatchFilter:
    """
    Callable class that filters prompts for an exact match from a list of allowed classes.
    This is picklable and safe for multiprocessing.
    """
    def __init__(self, allowed_classes: typing.List[str]):
        """
        Args:
            allowed_classes: A list of strings representing the prompts to keep.
        """
        self.allowed_set = set(allowed_classes)

    def __call__(self, text: str) -> bool:
        """
        Returns True if the text is in the allowed set.
        """
        return text in self.allowed_set

def create_locate_classes_filter_function():
    return ExactMatchFilter(allowed_classes=LOCATE_CLASSES)

def create_babel_20_classes_filter_function():
    return ExactMatchFilter(allowed_classes=BABEL_20_CLASSES)

def create_babel_60_classes_filter_function():
    return ExactMatchFilter(allowed_classes=BABEL_60_CLASSES)

def create_babel_90_classes_filter_function():
    return ExactMatchFilter(allowed_classes=BABEL_90_CLASSES)

def create_babel_120_classes_filter_function():
    return ExactMatchFilter(allowed_classes=BABEL_120_CLASSES)

@dataclasses.dataclass
class FilterConfig:
    """
    Configuration for Babel filtering function.
    """
    seed: int = DEFAULT_SEED
    fps: int = DEFAULT_FPS
    min_motion_frames: int = 1
    max_motion_frames: int = 10000
    min_prompts_per_sample: int = 0
    max_prompts_per_sample: int = 100
    split_max_prompts_per_sample: bool = False
    prompt_text_filter_function: typing.Optional[typing.Callable[[str], bool]] = None
    min_span_frames: int = 1
    max_span_frames: int = 10000
    max_spans_per_prompt: int = 10
    debug: bool = False
    sources: list[str] = dataclasses.field(default_factory=lambda: ["proc_label"])

class FilterFunction:
    """
    Callable class to filter the Babel dataset based on a configuration.
    This is compatible with multiprocessing (num_workers > 0 in DataLoader).
    Works with datasets that have been processed with babel_simplify_batch_structure().
    """
    def __init__(self, config: FilterConfig):
        """
        Args:
            config (FilterConfig): Configuration object specifying all filtering parameters.
        """
        self.config = config
        # Seed in __init__ so that if the object is created once and passed to workers,
        # the main process's random state is set. Workers will inherit or have their own.
        random.seed(self.config.seed)

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        """
        Filters and transforms a batch of data.
        """
        # pdb.set_trace()
        new_batch = {key: [] for key in batch.keys()}
        num_samples_in = len(batch[next(iter(batch.keys()))])
        total_spans_filtered = 0
        total_prompts_filtered = 0
        samples_kept = 0
        
        for i in range(num_samples_in):
            sample_sid = batch.get("sid", ["N/A"])[i]
            prompts_list = batch["prompts"][i]
            
            # --- --- --- MOTION LENGTH FILTERING --- --- ---
            motion_length = len(batch["motion"][i]["new_joints"])
            if not (self.config.min_motion_frames <= motion_length <= self.config.max_motion_frames):
                if self.config.debug:
                    logger.debug(f"[Filter] SID {sample_sid}: Dropping sample, motion duration {motion_length} is outside [{self.config.min_motion_frames}, {self.config.max_motion_frames}].")
                continue
            
            # --- --- --- SPAN AND PROMPT FILTERING --- --- ---
            filtered_prompts_list = []
            spans_dropped_this_sample = 0
            
            # First, filter individual prompt records
            for prompt_data in prompts_list:
                prompt_text = prompt_data.get("text", "")
                span = prompt_data.get("span", [])
                source = prompt_data.get("source", "")
                is_sequence = prompt_data.get("is_sequence", True)
                
                if not span or not prompt_text or not source:
                    continue
                
                # Apply source filtering
                if source not in self.config.sources:
                    spans_dropped_this_sample += 1
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Dropping prompt '{prompt_text}' from source '{source}' (not in allowed sources: {self.config.sources}).")
                    continue
                
                # Apply prompt text filtering
                if self.config.prompt_text_filter_function and not self.config.prompt_text_filter_function(prompt_text):
                    spans_dropped_this_sample += 1
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Dropping prompt '{prompt_text}' due to custom filter function.")
                    continue
                
                # Filter spans by duration
                start_frame, end_frame = span
                duration = end_frame - start_frame
                if not (self.config.min_span_frames <= duration <= self.config.max_span_frames):
                    spans_dropped_this_sample += 1
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Dropping span for '{prompt_text}', duration {duration} is outside [{self.config.min_span_frames}, {self.config.max_span_frames}].")
                    continue
                
                # Keep this prompt
                filtered_prompts_list.append(prompt_data)
            
            # --- --- --- GROUP BY TEXT AND LIMIT SPANS PER PROMPT --- --- ---
            # Group prompts by text to handle max_spans_per_prompt
            prompts_by_text = {}
            for prompt_data in filtered_prompts_list:
                text = prompt_data["text"]
                if text not in prompts_by_text:
                    prompts_by_text[text] = []
                prompts_by_text[text].append(prompt_data)
            
            # Apply max_spans_per_prompt limit
            final_prompts_list = []
            for text, text_prompts in prompts_by_text.items():
                if len(text_prompts) > self.config.max_spans_per_prompt:
                    spans_dropped_this_sample += len(text_prompts) - self.config.max_spans_per_prompt
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Sampling {self.config.max_spans_per_prompt} from {len(text_prompts)} spans for prompt '{text}'.")
                    # Randomly sample spans
                    text_prompts = random.sample(text_prompts, self.config.max_spans_per_prompt)
                final_prompts_list.extend(text_prompts)
            
            # --- --- --- SAMPLE-LEVEL FILTERING --- --- ---
            # Count unique prompts
            unique_prompts = set(prompt_data["text"] for prompt_data in final_prompts_list)
            num_unique_prompts = len(unique_prompts)
            
            if num_unique_prompts < self.config.min_prompts_per_sample:
                total_spans_filtered += spans_dropped_this_sample
                total_prompts_filtered += len(final_prompts_list)  # All prompts in this sample are dropped
                if self.config.debug:
                    logger.debug(f"[Filter] SID {sample_sid}: Dropping sample, has {num_unique_prompts} prompts, less than min {self.config.min_prompts_per_sample}.")
                continue
            
            # Handle too many prompts
            if num_unique_prompts > self.config.max_prompts_per_sample:
                if self.config.split_max_prompts_per_sample:
                    # Split the sample: create multiple samples instead of discarding prompts
                    # Group prompts by text and then split groups
                    prompts_by_text_final = {}
                    for prompt_data in final_prompts_list:
                        text = prompt_data["text"]
                        if text not in prompts_by_text_final:
                            prompts_by_text_final[text] = []
                        prompts_by_text_final[text].append(prompt_data)
                    
                    # Create chunks of prompts (by unique text)
                    unique_texts = list(prompts_by_text_final.keys())
                    random.shuffle(unique_texts)  # Randomize the order
                    
                    # Split unique texts into chunks
                    text_chunks = []
                    for chunk_start in range(0, len(unique_texts), self.config.max_prompts_per_sample):
                        chunk_end = min(chunk_start + self.config.max_prompts_per_sample, len(unique_texts))
                        text_chunks.append(unique_texts[chunk_start:chunk_end])
                    
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Splitting sample into {len(text_chunks)} samples with {[len(chunk) for chunk in text_chunks]} unique prompts each.")
                    
                    # Create a sample for each chunk
                    for chunk_idx, chunk_texts in enumerate(text_chunks):
                        # Collect all prompts for this chunk
                        chunk_prompts = []
                        for text in chunk_texts:
                            chunk_prompts.extend(prompts_by_text_final[text])
                        
                        # Add this chunk as a new sample
                        for key in batch.keys():
                            if key == "prompts":
                                new_batch[key].append(chunk_prompts)
                            else:
                                new_batch[key].append(batch[key][i])
                        samples_kept += 1
                    
                    total_spans_filtered += spans_dropped_this_sample
                    continue  # Skip the normal single sample processing
                else:
                    # Original behavior: randomly sample and discard excess prompts
                    total_prompts_filtered += num_unique_prompts - self.config.max_prompts_per_sample
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Sampling {self.config.max_spans_per_prompt} from {num_unique_prompts} unique prompts.")
                    
                    # Group prompts by text and randomly sample texts
                    prompts_by_text_final = {}
                    for prompt_data in final_prompts_list:
                        text = prompt_data["text"]
                        if text not in prompts_by_text_final:
                            prompts_by_text_final[text] = []
                        prompts_by_text_final[text].append(prompt_data)
                    
                    # Randomly sample unique texts
                    unique_texts = list(prompts_by_text_final.keys())
                    sampled_texts = random.sample(unique_texts, self.config.max_prompts_per_sample)
                    
                    # Collect all prompts for sampled texts
                    final_prompts_list = []
                    for text in sampled_texts:
                        final_prompts_list.extend(prompts_by_text_final[text])
            
            # Add the filtered sample
            for key in batch.keys():
                if key == "prompts":
                    new_batch[key].append(final_prompts_list)
                else:
                    new_batch[key].append(batch[key][i])
            
            total_spans_filtered += spans_dropped_this_sample
            samples_kept += 1
        
        samples_dropped = num_samples_in - samples_kept
        print(f"[Filter] Processing complete:")
        print(f"\tSamples: {samples_kept} kept, {samples_dropped} dropped (out of {num_samples_in} total)")
        print(f"\tTotal spans filtered out: {total_spans_filtered}")
        print(f"\tTotal prompts filtered out: {total_prompts_filtered}")
        
        return new_batch