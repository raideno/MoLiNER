# import sys

# if "../" not in sys.path:
#     sys.path.append("../")
#     print("[sys.path]:", sys.path)
    
# --- --- --- xxx --- --- ---

import os
import pdb
import tqdm
import torch
import datasets

import numpy as np

from src.data.babel import BabelDataset
from src.auth import login_to_huggingface
from src.data.utils.batching import babel_simplify_batch_structure

from src.constants import (
    HUGGING_FACE_TOKEN,
    BABEL_REMOTE_DATASET_NAME,
    HML3D_REMOTE_DATASET_NAME,
)

from src.data.babel import BabelDataset

def main():
    login_to_huggingface()
    
    dataset = BabelDataset(
        split="validation",
        pipeline="locate",
    )
    
    pdb.set_trace()
    
    print("[#dataset]:", len(dataset))
    
    # babel_dataset = datasets.load_dataset(
    #     BABEL_REMOTE_DATASET_NAME,
    #     trust_remote_code=True,
    #     name="full_all_motion"
    # )
    
    # from src.data.utils.batching import babel_simplify_batch_structure
        
    # pdb.set_trace()

    # clean_dataset = babel_dataset["validation"].map(
    #     babel_simplify_batch_structure,
    #     load_from_cache_file=False
    # )
    
    # pdb.set_trace()
    
    # print("[#clean_dataset]:", len(clean_dataset))
    
    # from src.data.utils.filtering import FilterConfig, FilterFunction, create_locate_classes_filter_function
    
    # filter_config = FilterConfig(
    #     seed=42,
    #     fps=20,
    #     prompt_text_filter_function=create_locate_classes_filter_function(),
    #     min_motion_frames=4,         # At least 10 frames (0.5 seconds at 20fps)
    #     max_motion_frames=512,        # At most 600 frames (30 seconds at 20fps)
    #     min_prompts_per_sample=1,     # At least 1 prompt per sample
    #     max_prompts_per_sample=16,    # At most 10 unique prompts per sample
    #     split_max_prompts_per_sample=False,  # Don't split samples with too many prompts
    #     min_span_frames=4,            # Spans must be at least 4 frames
    #     max_span_frames=128,          # Spans must be at most 128 frames
    #     max_spans_per_prompt=8,       # Max 3 spans per unique prompt text
    #     sources=["act_cat"],  # Only keep prompts from these sources
    #     debug=True                    # Enable debug logging
    # )
    
    # filtered_dataset = clean_dataset.map(
    #     FilterFunction(filter_config),
    #     batched=True,
    #     batch_size=16
    # )
    
    # pdb.set_trace()

    # print("[#filtered_dataset]:", len(clean_dataset))

    
if __name__ == "__main__":
    main()