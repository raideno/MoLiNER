import enum
import torch
import typing
import random

from src.data.typing import RawBatch

def _normalize_annotations(annotations: dict) -> list[dict]:
    """
    Transform column format annotations to row format.
    """
    if "labels" not in annotations or not annotations["labels"]:
        return []
    
    labels = annotations["labels"]
    # NOTE: already in row format
    if isinstance(labels, list):
        return labels
    
    # NOTE: in column format
    if isinstance(labels, dict):
        # NOTE: we transpose
        keys = labels.keys()
        values_per_key = labels.values()
        num_labels = len(next(iter(values_per_key), []))
        
        reformatted_labels = []
        for i in range(num_labels):
            # print(i, keys, labels)
            reformatted_labels.append({key: labels[key][i] for key in keys})
        return reformatted_labels
        
    return []

from src.constants import (
    DEFAULT_FPS,
    DEFAULT_PADDING_VALUE
)

def babel_simplify_batch_structure(sample: dict) -> dict:
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

def hml3d_simplify_batch_structure(batch: dict[str, list]) -> dict[str, list]:
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

def babel_augment_and_split_batch(batch: dict[str, list]) -> dict[str, list]:
    """
    A function for `datasets.map(batched=True)` to augment the Babel dataset.
    Will split samples with both sequence and frame annotations into two distinct samples.
    Works with the simplified prompt structure where prompts are individual records.
    """
    new_batch = {key: [] for key in batch.keys()}
    
    num_samples = len(batch[next(iter(batch.keys()))])
    
    for i in range(num_samples):
        prompts_list = batch["prompts"][i]
        
        has_seq_prompts = any(prompt_data.get("is_sequence", True) for prompt_data in prompts_list)
        has_frame_prompts = any(not prompt_data.get("is_sequence", True) for prompt_data in prompts_list)
        
        if has_seq_prompts and has_frame_prompts:
            sequence_prompts = [prompt_data for prompt_data in prompts_list if prompt_data.get("is_sequence", True)]
            
            for key in batch.keys():
                if key == "prompts":
                    new_batch[key].append(sequence_prompts)
                else:
                    new_batch[key].append(batch[key][i])
            
            frame_prompts = [prompt_data for prompt_data in prompts_list if not prompt_data.get("is_sequence", True)]
            
            for key in batch.keys():
                if key == "prompts":
                    new_batch[key].append(frame_prompts)
                else:
                    new_batch[key].append(batch[key][i])
        else:
            # NOTE: if sample has only one type of prompts (or no prompts), we keept it as is
            for key in batch.keys():
                new_batch[key].append(batch[key][i])
    
    return new_batch

class SimplifiedBabelCollateFn:
    def __init__(self):
        self.fps = DEFAULT_FPS
        self.padding_value = DEFAULT_PADDING_VALUE

    def __call__(self, batch: list[dict]) -> "RawBatch":
        if not all(isinstance(sample.get("motion"), dict) for sample in batch):
            raise ValueError("A sample's 'motion' field is not a dictionary. Please load the dataset with a configuration like 'full_all_motion'.")
        if not all("new_joints" in sample["motion"] and "new_joint_vecs" in sample["motion"] for sample in batch):
            raise ValueError("A sample is missing 'new_joints' or 'new_joint_vecs'. Please use the 'full_all_motion' configuration.")

        joints_list = [torch.tensor(sample["motion"]["new_joints"], dtype=torch.float32) for sample in batch]
        vecs_list = [torch.tensor(sample["motion"]["new_joint_vecs"], dtype=torch.float32) for sample in batch]
        lengths = [motion.shape[0] for motion in joints_list]

        padded_joints = torch.nn.utils.rnn.pad_sequence(joints_list, batch_first=True, padding_value=self.padding_value)
        padded_vecs = torch.nn.utils.rnn.pad_sequence(vecs_list, batch_first=True, padding_value=self.padding_value)

        mask = torch.zeros(padded_joints.shape[:2], dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        sids: list[int] = [sample["sid"] for sample in batch]
        amass_paths: list[str] = [sample["amass_file_relative_path"] for sample in batch]
        dataset_names: list[str] = ["babel"] * len(batch)

        all_prompts = []
        for i, sample in enumerate(batch):
            sample_prompts = []
            prompts_list = sample.get("prompts", [])
            
            # NOTE: convert the prompts list to the expected format
            # From: [{"text": str, "span": [start, end], "source": str, "is_sequence": bool}, ...]
            # To: [(prompt_text, [(start_frame, end_frame), ...], is_sequence_prompt), ...]
            
            # NOTE: group prompts by text to consolidate spans for the same prompt
            prompts_dict = {}
            for prompt_data in prompts_list:
                prompt_text = prompt_data.get("text", "")
                span = prompt_data.get("span", [])
                is_sequence = prompt_data.get("is_sequence", True)
                
                if prompt_text and span:
                    if prompt_text not in prompts_dict:
                        prompts_dict[prompt_text] = {
                            "spans": [],
                            "is_sequence": is_sequence
                        }
                    prompts_dict[prompt_text]["spans"].append((span[0], span[1]))
            
            for prompt_text, data in prompts_dict.items():
                sample_prompts.append((prompt_text, data["spans"], data["is_sequence"]))
            
            all_prompts.append(sample_prompts)

        return RawBatch(
            sid=sids,
            dataset_name=dataset_names,
            amass_relative_path=amass_paths,
            raw_motion=padded_joints,
            transformed_motion=padded_vecs,
            motion_mask=mask,
            prompts=all_prompts
        )