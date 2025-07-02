import random
import typing
from dataclasses import dataclass

from src.constants import DEFAULT_FPS, DEFAULT_SEED

from src.constants import (
    BABEL_20_CLASSES,
    BABEL_60_CLASSES,
    BABEL_90_CLASSES,
    BABEL_120_CLASSES,
    LOCATE_CLASSES,
)

@dataclass
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
    prompt_text_filter_fn: typing.Optional[typing.Callable[[str], bool]] = None
    min_span_frames: int = 1
    max_span_frames: int = 10000
    max_spans_per_prompt: int = 10
    debug: bool = False

def create_babel_filter_fn(config: FilterConfig):
    random.seed(config.seed)

    def filter_and_transform_batch(batch: dict[str, list]) -> dict[str, list]:
        new_batch = {key: [] for key in batch.keys()}
        num_samples_in = len(batch[next(iter(batch.keys()))])
        
        total_spans_filtered = 0
        total_prompts_filtered = 0
        samples_kept = 0

        for i in range(num_samples_in):
            sample_sid = batch.get("sid", ["N/A"])[i]
            
            def get_spans_from_annotation_dict(ann_dict, source_name):
                spans = []
                labels = ann_dict.get("labels", {})
                if not labels:
                    return []
                
                num_labels = len(next(iter(labels.values()), []))
                for k in range(num_labels):
                    span_data = {key: labels[key][k] for key in labels}
                    span_data["source"] = source_name
                    spans.append(span_data)
                return spans

            seq_ann = batch["sequence_annotations"][i]
            frame_ann = batch["frame_annotations"][i]
            
            all_spans = get_spans_from_annotation_dict(seq_ann, 'sequence') + \
                        get_spans_from_annotation_dict(frame_ann, 'frame')

            motion_len = len(batch["motion"][i]["new_joints"])
            if not (config.min_motion_frames <= motion_len <= config.max_motion_frames):
                if config.debug:
                    print(f"[Filter] SID {sample_sid}: Dropping sample, motion duration {motion_len} is outside [{config.min_motion_frames}, {config.max_motion_frames}].")
                continue

            filtered_spans = []
            spans_dropped_this_sample = 0
            for span in all_spans:
                start_frame = int(span.get("start_t", 0) * config.fps)
                end_frame = min(int(span.get("end_t", 0) * config.fps), motion_len - 1)
                duration = end_frame - start_frame
                
                if not (config.min_span_frames <= duration <= config.max_span_frames):
                    spans_dropped_this_sample += 1
                    if config.debug:
                         print(f"[Filter] SID {sample_sid}: Dropping span '{span['proc_label']}', duration {duration} is outside [{config.min_span_frames}, {config.max_span_frames}].")
                    continue
                
                text = span.get("proc_label", "")
                # if any(b_word in text for b_word in prompt_text_blacklist):
                #     if debug:
                #         print(f"[Filter] SID {sample_sid}: Dropping span '{text}' due to blacklist.")
                #     continue
                if config.prompt_text_filter_fn and not config.prompt_text_filter_fn(text):
                    spans_dropped_this_sample += 1
                    if config.debug:
                        print(f"[Filter] SID {sample_sid}: Dropping span '{text}' due to custom filter function.")
                    continue
                filtered_spans.append(span)
            
            prompts_by_text = {}
            for span in filtered_spans:
                text = span.get("proc_label")
                if text:
                    prompts_by_text.setdefault(text, []).append(span)

            sampled_prompts = {}
            prompts_dropped_this_sample = 0
            for text, spans in prompts_by_text.items():
                if len(spans) > config.max_spans_per_prompt:
                    prompts_dropped_this_sample += len(spans) - config.max_spans_per_prompt
                    if config.debug:
                        print(f"[Filter] SID {sample_sid}: Sampling {config.max_spans_per_prompt} from {len(spans)} spans for prompt '{text}'.")
                    sampled_prompts[text] = random.sample(spans, config.max_spans_per_prompt)
                else:
                    sampled_prompts[text] = spans

            num_unique_prompts = len(sampled_prompts)
            if num_unique_prompts < config.min_prompts_per_sample:
                total_spans_filtered += spans_dropped_this_sample
                total_prompts_filtered += prompts_dropped_this_sample + num_unique_prompts  # All prompts in this sample are dropped
                if config.debug:
                    print(f"[Filter] SID {sample_sid}: Dropping sample, has {num_unique_prompts} prompts, less than min {config.min_prompts_per_sample}.")
                continue

            if num_unique_prompts > config.max_prompts_per_sample:
                prompts_dropped_this_sample += num_unique_prompts - config.max_prompts_per_sample
                if config.debug:
                    print(f"[Filter] SID {sample_sid}: Sampling {config.max_prompts_per_sample} from {num_unique_prompts} unique prompts.")
                keys_to_keep = random.sample(list(sampled_prompts.keys()), config.max_prompts_per_sample)
                sampled_prompts = {k: sampled_prompts[k] for k in keys_to_keep}
            
            final_spans_list = [span for spans in sampled_prompts.values() for span in spans]
            
            total_spans_filtered += spans_dropped_this_sample
            total_prompts_filtered += prompts_dropped_this_sample
            samples_kept += 1
            
            reconstructed_seq_labels = {key: [] for key in seq_ann.get("labels", {})}
            reconstructed_frame_labels = {key: [] for key in frame_ann.get("labels", {})}
            
            for span in final_spans_list:
                target_labels = reconstructed_seq_labels if span['source'] == 'sequence' else reconstructed_frame_labels
                for key in target_labels:
                    target_labels[key].append(span.get(key))

            for key in batch.keys():
                if key == "sequence_annotations":
                    new_batch[key].append({"labels": reconstructed_seq_labels})
                elif key == "frame_annotations":
                    new_batch[key].append({"labels": reconstructed_frame_labels})
                else:
                    new_batch[key].append(batch[key][i])
        
        samples_dropped = num_samples_in - samples_kept
        print(f"[Filter] Processing complete:")
        print(f"\tSamples: {samples_kept} kept, {samples_dropped} dropped (out of {num_samples_in} total)")
        print(f"\tTotal spans filtered out: {total_spans_filtered}")
        print(f"\tTotal prompts filtered out: {total_prompts_filtered}")
        
        return new_batch
    
    return filter_and_transform_batch

def no_transition_filter_fn_factory():
    def no_transition_filter_fn(text: str) -> bool:
        """
        Keeps the prompt if it does NOT contain any transition keywords.
        """
        transition_keywords = {"transition"}
        
        return not any(keyword in text for keyword in transition_keywords)

def create_exact_match_filter(
    allowed_classes: typing.List[str]
) -> typing.Callable[[str], bool]:
    """
    Factory to create a filter that only keeps prompts that are an exact match to one of the strings in the provided list.

    This is ideal for filtering based on predefined class sets like LOCATE or BABEL-X.

    Args:
        allowed_classes: A list of strings representing the entire set of
                         prompts to keep.

    Returns:
        A callable that returns `True` if the prompt is in the allowed set, `False` otherwise.
    """
    allowed_set = set(allowed_classes)

    def filter_fn(text: str) -> bool:
        """
        Keeps the prompt if its text is an exact match in the allowed set.
        """
        return text in allowed_set
        
    return filter_fn

def create_locate_classes_filter_fn():
    return create_exact_match_filter(allowed_classes=LOCATE_CLASSES)

def create_babel_20_classes_filter_fn():
    return create_exact_match_filter(allowed_classes=BABEL_20_CLASSES)

def create_babel_60_classes_filter_fn():
    return create_exact_match_filter(allowed_classes=BABEL_60_CLASSES)

def create_babel_90_classes_filter_fn():
    return create_exact_match_filter(allowed_classes=BABEL_90_CLASSES)

def create_babel_120_classes_filter_fn():
    return create_exact_match_filter(allowed_classes=BABEL_120_CLASSES)
