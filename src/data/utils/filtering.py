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
    def __init__(self):
        self.transition_keywords = {"transition"}

    def __call__(self, text: str) -> bool:
        return not any(keyword in text for keyword in self.transition_keywords)

class ExactMatchFilter:
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

# TODO: the "min_prompts_per_sample" property should be used to filter on the unique prompts level rather than the total. 
# TODO: for some filtering properties, we should have the option to either drop the sample if it doesn't meet the criteria, or modify it to meet the criteria.
@dataclasses.dataclass
class FilterConfig:
    """
    Configuration for filtering function.
    All fields are optional - filtering is only applied when values are specified.
    
    Args:
        seed: Random seed for reproducible sampling operations.
        
        fps: Frames per second (currently unused in filtering logic).
        
        min_motion_frames: Minimum number of motion frames required.
            Behavior: DROPS entire sample if motion has fewer frames.
            
        max_motion_frames: Maximum number of motion frames allowed.
            Behavior: DROPS entire sample if motion has more frames.
            
        min_prompts_per_sample: Minimum number of unique prompts required per sample.
            Behavior: DROPS entire sample if it has fewer unique prompts after all other filtering.
            
        max_prompts_per_sample: Maximum number of unique prompts allowed per sample.
            Behavior: MODIFIES sample by randomly sampling prompts to meet limit, OR
                     SPLITS sample into multiple samples if split_max_prompts_per_sample=True.
                     
        split_max_prompts_per_sample: Controls behavior when max_prompts_per_sample is exceeded.
            - False: MODIFIES sample by randomly sampling to meet limit
            - True: SPLITS sample into multiple samples, each meeting the limit
            
        prompt_text_filter_function: Custom function to filter prompts by text content.
            Behavior: DROPS individual prompts/spans that don't pass the filter function.
            
        min_span_frames: Minimum number of frames required for individual spans.
            Behavior: DROPS individual spans that are too short.
            
        max_span_frames: Maximum number of frames allowed for individual spans.
            Behavior: DROPS individual spans that are too long.
            
        min_spans_per_prompt: Minimum number of spans required per unique prompt text.
            Behavior: DROPS all spans for a prompt if it has fewer than required spans.
            
        max_spans_per_prompt: Maximum number of spans allowed per unique prompt text.
            Behavior: MODIFIES prompt by randomly sampling spans to meet limit.
            
        debug: Enable debug logging to track filtering decisions.
        
        sources: List of allowed annotation sources (e.g., ["proc_label", "act_cat"]).
            Behavior: DROPS individual prompts/spans from disallowed sources.
            
        annotation_types: List of annotation types to keep. Can contain "frames", "sequence", or both.
            - ["frames"]: Keep only frame annotations (is_sequence=False)
            - ["sequence"]: Keep only sequence annotations (is_sequence=True)  
            - ["frames", "sequence"]: Keep both types (default behavior)
            - None: Keep both types (default behavior)
            - Invalid values (not "frames" or "sequence") are ignored with a warning
            Behavior: DROPS individual prompts/spans that don't match allowed types.
    """
    seed: typing.Optional[int] = DEFAULT_SEED
    fps: typing.Optional[int] = DEFAULT_FPS
    
    min_motion_frames: typing.Optional[int] = None
    max_motion_frames: typing.Optional[int] = None
    
    min_prompts_per_sample: typing.Optional[int] = None
    max_prompts_per_sample: typing.Optional[int] = None
    
    split_max_prompts_per_sample: bool = False
    
    prompt_text_filter_function: typing.Optional[typing.Callable[[str], bool]] = None
    
    min_span_frames: typing.Optional[int] = None
    max_span_frames: typing.Optional[int] = None
    
    min_spans_per_prompt: typing.Optional[int] = None
    max_spans_per_prompt: typing.Optional[int] = None
    
    debug: bool = False
    
    sources: typing.Optional[list[str]] = None
    annotation_types: typing.Optional[list[str]] = None

class FilterFunction:
    """
    Callable class to filter the dataset based on a configuration.
    Works with datasets that have been processed with <dataset>_simplify_batch_structure().
    """
    def __init__(self, config: FilterConfig):
        """
        Args:
            config (FilterConfig): Configuration object specifying all filtering parameters.
        """
        self.config = config
       
        if self.config.seed is not None:
            random.seed(self.config.seed)

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        new_batch = {key: [] for key in batch.keys()}
        num_samples_in = len(batch[next(iter(batch.keys()))])
        
        total_spans_filtered = 0
        total_unique_prompts_filtered = 0
        samples_kept = 0
        
        for i in range(num_samples_in):
            sample_sid = batch.get("sid", ["N/A"])[i]
            prompts_list = batch["prompts"][i]
            
            # NOTE: motion length filtering
            motion_length = len(batch["motion"][i]["new_joints"])
            if (self.config.min_motion_frames is not None and motion_length < self.config.min_motion_frames) or \
               (self.config.max_motion_frames is not None and motion_length > self.config.max_motion_frames):
                if self.config.debug:
                    logger.debug(f"[Filter] SID {sample_sid}: Dropping sample, motion duration {motion_length} is outside limits.")
                continue
            
            # NOTE: span and prompt filtering
            filtered_prompts_list = []
            spans_dropped_this_sample = 0
            
            for prompt_data in prompts_list:
                prompt_text = prompt_data.get("text", "")
                span = prompt_data.get("span", [])
                source = prompt_data.get("source", "")

                                
                is_sequence = prompt_data.get("is_sequence", None)
                if is_sequence is None:
                    logger.warning(f"[Filter] SID {sample_sid}: Prompt '{prompt_text}' has no 'is_sequence' field, defaulting to True.")
                    is_sequence = True
                
                # TODO: careful about this as it'll filter out prompts with no span or text or source and we may not want this
                if not span or not prompt_text or not source:
                    continue
                
                # NOTE: source filtering
                if self.config.sources is not None and source not in self.config.sources:
                    spans_dropped_this_sample += 1
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Dropping prompt '{prompt_text}' from source '{source}' (not in allowed sources: {self.config.sources}).")
                    continue
                
                # NOTE: annotation type (frame / sequence) filtering
                if self.config.annotation_types is not None:
                    allowed_types = set(self.config.annotation_types)
                    if "frames" in allowed_types and "sequence" in allowed_types:
                        # NOTE: keep both types - no filtering needed
                        pass
                    elif "frames" in allowed_types and is_sequence:
                        # NOTE: only keeping frames but this is a sequence annotation - drop it
                        spans_dropped_this_sample += 1
                        if self.config.debug:
                            logger.debug(f"[Filter] SID {sample_sid}: Dropping sequence annotation '{prompt_text}' (only frames allowed).")
                        continue
                    elif "sequence" in allowed_types and not is_sequence:
                        # NOTE: only keeping sequences but this is a frame annotation - drop it
                        spans_dropped_this_sample += 1
                        if self.config.debug:
                            logger.debug(f"[Filter] SID {sample_sid}: Dropping frame annotation '{prompt_text}' (only sequences allowed).")
                        continue
                    elif len(allowed_types) == 0 or not (allowed_types.intersection({"frames", "sequence"})):
                        # NOTE: invalid annotation_types specified; won't be filtered
                        # TODO: maybe we should crash
                        logger.warning(f"[Filter] Invalid annotation_types specified: {self.config.annotation_types}. Should contain 'frames' and/or 'sequence'.")
                
                # NOTE: prompt text filtering
                if self.config.prompt_text_filter_function and not self.config.prompt_text_filter_function(prompt_text):
                    spans_dropped_this_sample += 1
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Dropping prompt '{prompt_text}' due to custom filter function.")
                    continue
                
                # NOTE: span duration filtering
                start_frame, end_frame = span
                duration = end_frame - start_frame
                if (self.config.min_span_frames is not None and duration < self.config.min_span_frames) or \
                   (self.config.max_span_frames is not None and duration > self.config.max_span_frames):
                    spans_dropped_this_sample += 1
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Dropping span for '{prompt_text}', duration {duration} is outside limits.")
                    continue
                
                filtered_prompts_list.append(prompt_data)
            
            # NOTE: group prompts by text and limit spans per prompt
            prompts_by_text = {}
            for prompt_data in filtered_prompts_list:
                text = prompt_data["text"]
                if text not in prompts_by_text:
                    prompts_by_text[text] = []
                prompts_by_text[text].append(prompt_data)
            
            final_prompts_list = []
            for text, text_prompts in prompts_by_text.items():
                # NOTE: min_spans_per_prompt filtering
                if self.config.min_spans_per_prompt is not None and len(text_prompts) < self.config.min_spans_per_prompt:
                    spans_dropped_this_sample += len(text_prompts)
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Dropping all {len(text_prompts)} spans for prompt '{text}' (minimum {self.config.min_spans_per_prompt} required).")
                    continue
                
                if self.config.max_spans_per_prompt is not None and len(text_prompts) > self.config.max_spans_per_prompt:
                    max_spans = self.config.max_spans_per_prompt
                    spans_dropped_this_sample += len(text_prompts) - max_spans
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Sampling {max_spans} from {len(text_prompts)} spans for prompt '{text}'.")
                    # NOTE: randomly sample spans to keep
                    text_prompts = random.sample(text_prompts, max_spans)
                final_prompts_list.extend(text_prompts)
            
            # NOTE: sample-level filtering
            unique_prompts = set(prompt_data["text"] for prompt_data in final_prompts_list)
            num_unique_prompts = len(unique_prompts)
            
            if self.config.min_prompts_per_sample is not None and num_unique_prompts < self.config.min_prompts_per_sample:
                total_spans_filtered += spans_dropped_this_sample
                total_unique_prompts_filtered += num_unique_prompts
                if self.config.debug:
                    logger.debug(f"[Filter] SID {sample_sid}: Dropping sample, has {num_unique_prompts} prompts, less than min {self.config.min_prompts_per_sample}.")
                continue
            
            # NOTE: handle too many prompts
            if self.config.max_prompts_per_sample is not None and num_unique_prompts > self.config.max_prompts_per_sample:
                if self.config.split_max_prompts_per_sample:
                    # NOTE: split the sample - create multiple samples instead of discarding prompts
                    prompts_by_text_final = {}
                    for prompt_data in final_prompts_list:
                        text = prompt_data["text"]
                        if text not in prompts_by_text_final:
                            prompts_by_text_final[text] = []
                        prompts_by_text_final[text].append(prompt_data)
                    
                    # NOTE: create chunks of prompts (by unique text)
                    unique_texts = list(prompts_by_text_final.keys())
                    random.shuffle(unique_texts)
                    
                    # NOTE: split unique texts into chunks
                    text_chunks = []
                    max_prompts = self.config.max_prompts_per_sample
                    assert max_prompts is not None
                    for chunk_start in range(0, len(unique_texts), max_prompts):
                        chunk_end = min(chunk_start + max_prompts, len(unique_texts))
                        text_chunks.append(unique_texts[chunk_start:chunk_end])
                    
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Splitting sample into {len(text_chunks)} samples with {[len(chunk) for chunk in text_chunks]} unique prompts each.")
                    
                    # NOTE: create a sample for each chunk
                    for chunk_idx, chunk_texts in enumerate(text_chunks):
                        # NOTE: collect all prompts for this chunk
                        chunk_prompts = []
                        for text in chunk_texts:
                            chunk_prompts.extend(prompts_by_text_final[text])
                        
                        # NOTE: add this chunk as a new sample
                        for key in batch.keys():
                            if key == "prompts":
                                new_batch[key].append(chunk_prompts)
                            else:
                                new_batch[key].append(batch[key][i])
                        samples_kept += 1
                    
                    total_spans_filtered += spans_dropped_this_sample
                    continue
                else:
                    # NOTE: original behavior - randomly sample and discard excess prompts
                    max_prompts = self.config.max_prompts_per_sample
                    assert max_prompts is not None
                    total_unique_prompts_filtered += num_unique_prompts - max_prompts
                    if self.config.debug:
                        logger.debug(f"[Filter] SID {sample_sid}: Sampling {max_prompts} from {num_unique_prompts} unique prompts.")
                    
                    # NOTE: group prompts by text and randomly sample texts
                    prompts_by_text_final = {}
                    for prompt_data in final_prompts_list:
                        text = prompt_data["text"]
                        if text not in prompts_by_text_final:
                            prompts_by_text_final[text] = []
                        prompts_by_text_final[text].append(prompt_data)
                    
                    # NOTE: randomly sample unique texts
                    unique_texts = list(prompts_by_text_final.keys())
                    max_prompts = self.config.max_prompts_per_sample
                    assert max_prompts is not None
                    sampled_texts = random.sample(unique_texts, max_prompts)
                    
                    # NOTE: collect all prompts for sampled texts
                    final_prompts_list = []
                    for text in sampled_texts:
                        final_prompts_list.extend(prompts_by_text_final[text])
            
            # NOTE: add the filtered sample
            for key in batch.keys():
                if key == "prompts":
                    new_batch[key].append(final_prompts_list)
                else:
                    new_batch[key].append(batch[key][i])
            
            total_spans_filtered += spans_dropped_this_sample
            samples_kept += 1
        
        samples_dropped = num_samples_in - samples_kept
        
        if self.config.debug:
            logger.info(f"[Filter] Processing complete:")
            logger.info(f"\tSamples: {samples_kept} kept, {samples_dropped} dropped (out of {num_samples_in} total)")
            logger.info(f"\tTotal spans filtered out: {total_spans_filtered}")
            logger.info(f"\tTotal unique prompts filtered out: {total_unique_prompts_filtered}")
        
        return new_batch
    
class HML3DRelativeLengthFilter:
    def __init__(
        self,
        max_relative_moment_length: float,
        debug: bool = False
    ):
        """
        Filters HumanML3D samples based on the relative length of target moments.

        Args:
            max_relative_moment_length (float): The maximum allowed ratio of target moment length to the total motion length. Defaults to 0.8 (80%).
            debug (bool): If True, logs detailed filtering decisions.
        """
        if not (0 < max_relative_moment_length <= 1.0):
            raise ValueError("max_relative_moment_length must be between 0 and 1.0")
        
        self.max_relative_moment_length = max_relative_moment_length
        self.debug = debug

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
        new_batch = {key: [] for key in batch.keys()}
        num_samples_in = len(batch[next(iter(batch.keys()))])

        samples_kept = 0
        samples_dropped = 0

        for i in range(num_samples_in):
            sample_sid = batch.get("sid", ["N/A"])[i]
            prompts_list = batch["prompts"][i]
            motion_data = batch["motion"][i]

            if not motion_data or "new_joints" not in motion_data:
                if self.debug:
                    logger.debug(f"[HML3DFilter] SID {sample_sid}: Skipping sample due to missing motion data or 'new_joints'.")
                continue

            motion_length_frames = len(motion_data["new_joints"])
            if motion_length_frames == 0:
                if self.debug:
                    logger.debug(f"[HML3DFilter] SID {sample_sid}: Skipping sample as motion has 0 frames.")
                continue

            sample_exceeds_limit = False
            if prompts_list:
                # NOTE: all prompts refer to the same span with different texts, consider only first one
                first_prompt = prompts_list[0]
                if "span" in first_prompt and first_prompt["span"]:
                    start_frame, end_frame = first_prompt["span"]
                    moment_duration_frames = end_frame - start_frame

                    if moment_duration_frames < 0:
                        moment_duration_frames = 0

                    relative_length = moment_duration_frames / motion_length_frames

                    if relative_length > self.max_relative_moment_length:
                        sample_exceeds_limit = True
                        if self.debug:
                            logger.debug(
                                f"[HML3DFilter] SID {sample_sid}: Sample exceeds limit. "
                                f"Motion frames: {motion_length_frames}, Moment frames: {moment_duration_frames} "
                                f"(relative: {relative_length:.2f} > {self.max_relative_moment_length})"
                            )
                else:
                    if self.debug:
                        logger.debug(f"[HML3DFilter] SID {sample_sid}: Skipping span length check as prompt has no valid 'span'.")
            else:
                if self.debug:
                    logger.debug(f"[HML3DFilter] SID {sample_sid}: Skipping span length check as sample has no prompts.")


            if sample_exceeds_limit:
                samples_dropped += 1
                # NOTE: we don't add the sample to the new batch; it get filtered
                continue
            else:
                for key in batch.keys():
                    new_batch[key].append(batch[key][i])
                samples_kept += 1

        if self.debug:
            logger.info(f"[HML3DFilter] Filtering complete: {samples_kept} samples kept, {samples_dropped} dropped (out of {num_samples_in} total).")

        return new_batch