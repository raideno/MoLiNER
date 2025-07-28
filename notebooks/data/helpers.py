import os
import pickle
import collections

# TODO: add src path in case it isn't already set
from src.constants import (
    DEFAULT_FPS
)
# import Counter, defaultdict

def analyze_prompts(
    dataset,
    dataset_name: str,
    split_name: str,
):
    all_prompts = []
    sample_prompt_counts = []
    source_counts = collections.Counter()
    is_sequence_counts = collections.Counter()
    durations = []
    sequence_durations = []
    frame_durations = []
    spans_per_prompt = collections.defaultdict(list)
    
    motion_lengths_frames = []
    motion_lengths_seconds = []
    spans_per_motion = []
    spans_durations_per_motion = []
    
    for i, sample in enumerate(dataset):
        prompts_list = sample.get("prompts", [])
        sample_prompt_counts.append(len(prompts_list))
        
        motion = sample.get("motion", {})
        if isinstance(motion, dict) and "new_joint_vecs" in motion:
            motion_length = len(motion["new_joint_vecs"])
            motion_lengths_frames.append(motion_length)
            motion_lengths_seconds.append(motion_length / DEFAULT_FPS)
        
        motion_spans = []
        motion_spans_durations = []
        
        for prompt_data in prompts_list:
            prompt_text = prompt_data.get("text", "")
            span = prompt_data.get("span", [])
            source = prompt_data.get("source", "unknown")
            is_sequence = prompt_data.get("is_sequence", True)
            
            all_prompts.append(prompt_text)
            source_counts[source] += 1
            is_sequence_counts[is_sequence] += 1
            
            if len(span) == 2:
                duration_frames = span[1] - span[0]
                durations.append(duration_frames)
                spans_per_prompt[prompt_text].append(span)
                motion_spans.append(span)
                motion_spans_durations.append(duration_frames)
                
                if is_sequence:
                    sequence_durations.append(duration_frames)
                else:
                    frame_durations.append(duration_frames)
        
        spans_per_motion.append(len(motion_spans))
        spans_durations_per_motion.append(sum(motion_spans_durations) if motion_spans_durations else 0)
    
    return {
        'all_prompts': all_prompts,
        'sample_prompt_counts': sample_prompt_counts,
        'source_counts': source_counts,
        'is_sequence_counts': is_sequence_counts,
        'durations': durations,
        'sequence_durations': sequence_durations,
        'frame_durations': frame_durations,
        'spans_per_prompt': spans_per_prompt,
        'split_name': split_name,
        'dataset_name': dataset_name,
        'motion_lengths_frames': motion_lengths_frames,
        'motion_lengths_seconds': motion_lengths_seconds,
        'spans_per_motion': spans_per_motion,
        'spans_durations_per_motion': spans_durations_per_motion
    }
    
DEFAULT_CACHE_DIR = "./cache.local"

def get_analysis_with_cache(
    dataset,
    dataset_name: str,
    pipeline_name: str,
    split_name: str,
    cache_dir=DEFAULT_CACHE_DIR
):
    os.makedirs(cache_dir, exist_ok=True)
    fingerprint = getattr(dataset, 'fingerprint', None)
    
    cache_path = os.path.join(cache_dir, f"{dataset_name}_{pipeline_name}_{split_name.lower()}_{fingerprint}.pkl")
    
    if fingerprint and os.path.exists(cache_path):
        print(f"[cache]: Loading {split_name} analysis from {cache_path}")
        with open(cache_path, "rb") as file:
            return pickle.load(file)
    else:
        print(f"[cache]: Computing {split_name} analysis and saving to {cache_path}")
        analysis = analyze_prompts(
            dataset=dataset,
            dataset_name=dataset_name,
            split_name=split_name
        )
        if fingerprint:
            with open(cache_path, "wb") as file:
                pickle.dump(analysis, file)
        return analysis
