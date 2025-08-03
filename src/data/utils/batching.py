import typing

from src.types import DatasetSample
from src.data.utils.helpers import _normalize_annotations
from src.constants import (
    DEFAULT_FPS
)
    
def babel_simplify_batch_structure(sample: dict) -> DatasetSample:
    """
    A function for `datasets.map(batched=False)` to simplify the Babel dataset structure.
    Consolidates sequence_annotations and frame_annotations into a single 'prompts' field.
    
    Args:
        sample: Input sample with sequence_annotations and frame_annotations
        
    Returns:
        Modified sample with simplified structure where 'prompts' field contains:
        A list of dictionaries, each with:
        - "text": The prompt text
        - "span": [start_frame, end_frame] pair
        - "source": Source field name
        - "is_sequence": Boolean flag indicating if this is from sequence or frame annotations
    """
    from src.constants import DEFAULT_FPS
    
    new_sample = {key: value for key, value in sample.items() if key not in ["sequence_annotations", "frame_annotations"]}
    
    all_sources = ["proc_label", "raw_label", "act_cat"]
    
    motion_length = len(sample["motion"]["new_joints"])
    max_frame_idx = motion_length - 1
    
    sequence_annotations = sample["sequence_annotations"]
    sequence_labels = _normalize_annotations(sequence_annotations)
    
    frame_annotations = sample["frame_annotations"]
    frame_labels = _normalize_annotations(frame_annotations)
    
    prompts_list = []
    
    for label in sequence_labels:
        start_frame = int(label.get("start_t", 0) * DEFAULT_FPS)
        end_frame = min(int(label.get("end_t", 0) * DEFAULT_FPS), max_frame_idx)
        
        for source in all_sources:
            prompt_text = label.get(source)
            
            if prompt_text:
                if isinstance(prompt_text, list):
                    for text in prompt_text:
                        if text:
                            prompts_list.append({
                                "text": text,
                                "span": [start_frame, end_frame],
                                "source": source,
                                "is_sequence": True
                            })
                else:
                    prompts_list.append({
                        "text": prompt_text,
                        "span": [start_frame, end_frame],
                        "source": source,
                        "is_sequence": True
                    })
    
    for label in frame_labels:
        start_frame = int(label.get("start_t", 0) * DEFAULT_FPS)
        end_frame = min(int(label.get("end_t", 0) * DEFAULT_FPS), max_frame_idx)
        
        for source in all_sources:
            prompt_text = label.get(source)
            
            if prompt_text:
                if isinstance(prompt_text, list):
                    for text in prompt_text:
                        if text:
                            prompts_list.append({
                                "text": text,
                                "span": [start_frame, end_frame],
                                "source": source,
                                "is_sequence": False
                            })
                else:
                    prompts_list.append({
                        "text": prompt_text,
                        "span": [start_frame, end_frame],
                        "source": source,
                        "is_sequence": False
                    })
    
    new_sample["prompts"] = prompts_list
    
    return new_sample

def hml3d_simplify_batch_structure(batch: dict[str, list]) -> DatasetSample:
    """
    A function for `datasets.map(batched=True)` to simplify the HML3D dataset structure.
    Converts the 'texts' field into a unified 'prompts' field compatible with the simplified structure.
    
    Args:
        batch: Input batch with 'texts' field containing list of strings
        fps: Frames per second for time-to-frame conversion
        max_texts: Maximum number of texts to sample per motion
        
    Returns:
        Modified batch with simplified structure where 'prompts' field contains:
        A list of dictionaries, each with:
        - "text": The prompt text
        - "spans": List of [start_frame, end_frame] pairs (one span covering the full motion)
        - "sources": List of source field names for each span (always "texts")
        - "is_sequence": List of boolean flags for each span (always True for HML3D)
    """
    new_batch = {key: [] for key in batch.keys()}
    
    new_batch["prompts"] = []
    
    num_samples = len(batch[next(iter(batch.keys()))])
    
    for i in range(num_samples):
        motion_length = len(batch["motion"][i]["new_joints"])
        max_frame_idx = motion_length - 1
        
        start_t = batch.get("start_t", [0.0] * num_samples)[i]
        end_t = batch.get("end_t", [0.0] * num_samples)[i]
        
        start_frame = int(start_t * DEFAULT_FPS)
        end_frame = min(int(end_t * DEFAULT_FPS), max_frame_idx)
        
        texts = batch["texts"][i]
        
        prompts_list = []
        for text in texts:
            if text:
                prompts_list.append({
                    "text": text,
                    "span": [start_frame, end_frame],
                    "source": "texts",
                    "is_sequence": True
                })
        
        for key in batch.keys():
            if key != "texts":
                new_batch[key].append(batch[key][i])
        
        new_batch["prompts"].append(prompts_list)
    
    if "texts" in new_batch:
        del new_batch["texts"]
    
    return new_batch