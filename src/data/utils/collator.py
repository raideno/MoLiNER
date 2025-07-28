import torch

from src.types import RawBatch

from src.constants import (
    DEFAULT_FPS,
    DEFAULT_PADDING_VALUE
)

class SimpleBatchStructureCollator:
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

        # NOTE: the None is required for the collator to work with a mixed dataset as the sids are dropped.
        sids: list = [sample.get("sid", None) for sample in batch]
        
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