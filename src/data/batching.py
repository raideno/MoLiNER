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

def babel_augment_and_split_batch(batch: dict[str, list]) -> dict[str, list]:
    """
    A function for `datasets.map(batched=True)` to augment the Babel dataset.
    Will split samples with both sequence and frame annotations into two distinct samples.
    """
    new_batch = {key: [] for key in batch.keys()}
    
    empty_annotation = {
        'labels': {
            'act_cat': [],
            'raw_label': [],
            'proc_label': [],
            'start_t': [],
            'end_t': [],
        }
    }

    num_samples = len(batch[next(iter(batch.keys()))])

    for i in range(num_samples):
        seq_ann = batch["sequence_annotations"][i]
        frame_ann = batch["frame_annotations"][i]

        has_seq_prompts = bool(_normalize_annotations(seq_ann))
        has_frame_prompts = bool(_normalize_annotations(frame_ann))

        if has_seq_prompts and has_frame_prompts:
            for key in batch.keys():
                if key == "frame_annotations":
                    new_batch[key].append(empty_annotation)
                else:
                    new_batch[key].append(batch[key][i])
            
            for key in batch.keys():
                if key == "sequence_annotations":
                    new_batch[key].append(empty_annotation)
                else:
                    new_batch[key].append(batch[key][i])
        else:
            for key in batch.keys():
                new_batch[key].append(batch[key][i])

    return new_batch

class PromptGenerationMode(enum.Enum):
    """
    Enum to specify which annotations to use for prompt generation.
    - `SEQUENCE_ANNOTATIONS`: Use only sequence annotations.
    - `FRAME_ANNOTATIONS`: Use only frame annotations.
    - `BOTH`: Use both sequence and frame annotations.
    - `FRAME_ANNOTATIONS_WITH_FALLBACK_TO_SEQUENCE`: Use frame annotations, but if they are empty, fall back to sequence annotations.
    """
    SEQUENCE_ANNOTATIONS = "SEQUENCE_ANNOTATIONS"
    FRAME_ANNOTATIONS = "FRAME_ANNOTATIONS"
    BOTH = "BOTH"
    FRAME_ANNOTATIONS_WITH_FALLBACK_TO_SEQUENCE = "SEQUENCE_ANNOTATIONS"

def babel_create_raw_batch_collate_fn(
    fps: int = 20,
    mode: PromptGenerationMode = PromptGenerationMode.SEQUENCE_ANNOTATIONS,
    padding_value: float = 0.0,
):
    """
    Factory to create a collate function for the Babel dataset that produces RawBatch objects.

    Args:
        fps (int): Frames per second to convert time annotations to frame indices.
        mode (PromptGenerationMode): Specifies which annotations to use for prompts.
        padding_value (float): The value to use for padding motion tensors.

    Returns:
        A collate function that takes a batch and returns a RawBatch object.
    """
    
    if isinstance(mode, str):
        mode = PromptGenerationMode(mode)
    
    def babel_to_raw_batch(batch: list[dict]) -> "RawBatch":
        if not all(isinstance(sample.get("motion"), dict) for sample in batch):
             raise ValueError(
                "A sample's 'motion' field is not a dictionary. "
                "Please load the dataset with a configuration like 'full_all_motion'."
             )
        if not all("new_joints" in sample["motion"] and "new_joint_vecs" in sample["motion"] for sample in batch):
            raise ValueError(
                "A sample is missing 'new_joints' or 'new_joint_vecs'. "
                "Please use the 'full_all_motion' configuration."
            )

        joints_list = [torch.tensor(sample["motion"]["new_joints"], dtype=torch.float32) for sample in batch]
        vecs_list = [torch.tensor(sample["motion"]["new_joint_vecs"], dtype=torch.float32) for sample in batch]
        
        lengths = [motion.shape[0] for motion in joints_list]

        padded_joints = torch.nn.utils.rnn.pad_sequence(joints_list, batch_first=True, padding_value=padding_value)
        padded_vecs = torch.nn.utils.rnn.pad_sequence(vecs_list, batch_first=True, padding_value=padding_value)

        mask = torch.zeros(padded_joints.shape[:2], dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        sids: list[int] = [sample["sid"] for sample in batch]
        amass_paths: list[str] = [sample["amass_file_relative_path"] for sample in batch]
        dataset_names: list[str] = ["babel"] * len(batch)
        
        def get_prompts_from_annotations(annotations: typing.Dict[str, typing.Any], max_frame: int, sequence_prompt: bool = True) -> typing.List[typing.Tuple[str, int, int, bool]]:
            """
            Helper to extract and format prompts from a given annotation dict.
            Returns individual (text, start_frame, end_frame, is_sequence_prompt) tuples.
            """
            prompts = []
            normalized_labels = _normalize_annotations(annotations)
            for label in normalized_labels:
                start_frame = int(label.get("start_t", 0) * fps)
                end_frame = min(int(label.get("end_t", 0) * fps), max_frame)
                prompt_text = label.get("proc_label")
                
                if prompt_text:
                    prompts.append((prompt_text, start_frame, end_frame, True, sequence_prompt))
                    
            return prompts
        
        def group_prompts_by_text(prompts_list: typing.List[typing.Tuple[str, int, int, bool, bool]]) -> typing.List[typing.Tuple[str, typing.List[typing.Tuple[int, int]], bool, bool]]:
            """
            Groups prompts with the same text together, combining their spans.
            """
            text_to_prompt = {}
            
            for text, start_frame, end_frame, is_positive, is_sequence_prompt in prompts_list:
                if text not in text_to_prompt:
                    text_to_prompt[text] = {
                        'spans': [],
                        'is_positive': is_positive,
                        'is_sequence_prompt': is_sequence_prompt
                    }
                text_to_prompt[text]['spans'].append((start_frame, end_frame))
            
            # Convert to the new format
            grouped_prompts = []
            for text, info in text_to_prompt.items():
                grouped_prompts.append((text, info['spans'], info['is_positive'], info['is_sequence_prompt']))
            
            return grouped_prompts

        all_prompts = []
        for i, sample in enumerate(batch):
            sample_prompts = []
            max_frame_idx = lengths[i] - 1
            
            seq_prompts = get_prompts_from_annotations(sample.get("sequence_annotations", {}), max_frame_idx, True)
            frame_prompts = get_prompts_from_annotations(sample.get("frame_annotations", {}), max_frame_idx, False)

            if mode == PromptGenerationMode.BOTH:
                all_individual_prompts = seq_prompts + frame_prompts
            elif mode == PromptGenerationMode.SEQUENCE_ANNOTATIONS:
                all_individual_prompts = seq_prompts
            elif mode == PromptGenerationMode.FRAME_ANNOTATIONS:
                all_individual_prompts = frame_prompts
            elif mode == PromptGenerationMode.FRAME_ANNOTATIONS_WITH_FALLBACK_TO_SEQUENCE:
                all_individual_prompts = frame_prompts if frame_prompts else seq_prompts
            else:
                raise ValueError(f"Unsupported PromptGenerationMode: {mode}")

            # Group prompts by text to combine spans for the same prompt
            sample_prompts = group_prompts_by_text(all_individual_prompts)

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
        
    return babel_to_raw_batch

def hml3d_create_raw_batch_collate_fn(
    fps: int = 20,
    padding_value: float = 0.0,
    max_texts: int = 8
):
    """
    Factory to create a collate function for the HML3D dataset that produces RawBatch objects.

    Args:
        fps (int): Frames per second to convert time annotations to frame indices.
        padding_value (float): The value to use for padding motion tensors.

    Returns:
        A collate function that takes a batch and returns a RawBatch object.
    """
    
    def hml3d_to_raw_batch(batch: list[dict]) -> "RawBatch":
        """
        Collates a list of HML3D samples into a single RawBatch.
        Each sample in HML3D has a list of texts and one (start_t, end_t) pair that applies to the entire motion segment described.
        """
        if not all(isinstance(sample.get("motion"), dict) for sample in batch):
             raise ValueError(
                "A sample's 'motion' field is not a dictionary. "
                "Please load the dataset with a configuration like 'full_all_motion'."
             )
        if not all("new_joints" in sample["motion"] and "new_joint_vecs" in sample["motion"] for sample in batch):
            raise ValueError(
                "A sample is missing 'new_joints' or 'new_joint_vecs'. "
                "Please use the 'full_all_motion' configuration."
            )

        joints_list = [torch.tensor(sample["motion"]["new_joints"], dtype=torch.float32) for sample in batch]
        vecs_list = [torch.tensor(sample["motion"]["new_joint_vecs"], dtype=torch.float32) for sample in batch]
        
        lengths = [motion.shape[0] for motion in joints_list]

        padded_joints = torch.nn.utils.rnn.pad_sequence(joints_list, batch_first=True, padding_value=padding_value)
        padded_vecs = torch.nn.utils.rnn.pad_sequence(vecs_list, batch_first=True, padding_value=padding_value)

        mask = torch.zeros(padded_joints.shape[:2], dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = True

        sids: list[int] = [sample["sid"] for sample in batch]
        amass_paths: list[str] = [sample["amass_file_relative_path"] for sample in batch]
        dataset_names: list[str] = ["hml3d"] * len(batch)
        
        all_prompts = []
        for i, sample in enumerate(batch):
            sample_prompts = []
            max_frame_idx = lengths[i] - 1
            
            texts = sample.get("texts", [])

            start_t = sample.get("start_t", 0.0)
            end_t = sample.get("end_t", 0.0)

            start_frame = int(start_t * fps)
            end_frame = min(int(end_t * fps), max_frame_idx)

            selected_texts = random.sample(texts, min(len(texts), max_texts))

            for text in selected_texts:
                if text:
                    # Each text gets its own prompt with one span
                    prompt = (text, [(start_frame, end_frame)], True, True)
                    sample_prompts.append(prompt)
                    
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
        
    return hml3d_to_raw_batch

class BabelCollateFn:
    def __init__(
        self,
        fps: int = 20,
        mode: PromptGenerationMode = PromptGenerationMode.SEQUENCE_ANNOTATIONS,
        padding_value: float = 0.0,
    ):
        self.fps = fps
        self.mode = mode
        self.padding_value = padding_value
        
        if isinstance(self.mode, str):
            self.mode = PromptGenerationMode(self.mode)

    def _get_prompts_from_annotations(self, annotations: typing.Dict[str, typing.Any], max_frame: int, sequence_prompt: bool = True) -> typing.List[typing.Tuple[str, int, int, bool, bool]]:
        """Helper to extract and format prompts from a given annotation dict."""
        prompts = []
        normalized_labels = _normalize_annotations(annotations)
        for label in normalized_labels:
            start_frame = int(label.get("start_t", 0) * self.fps)
            end_frame = min(int(label.get("end_t", 0) * self.fps), max_frame)
            prompt_text = label.get("proc_label")
            
            if prompt_text:
                prompts.append((prompt_text, start_frame, end_frame, True, sequence_prompt))
                
        return prompts
    
    def _group_prompts_by_text(self, prompts_list: typing.List[typing.Tuple[str, int, int, bool, bool]]) -> typing.List[typing.Tuple[str, typing.List[typing.Tuple[int, int]], bool, bool]]:
        """
        Groups prompts with the same text together, combining their spans.
        """
        text_to_prompt = {}
        
        for text, start_frame, end_frame, is_positive, is_sequence_prompt in prompts_list:
            if text not in text_to_prompt:
                text_to_prompt[text] = {
                    'spans': [],
                    'is_positive': is_positive,
                    'is_sequence_prompt': is_sequence_prompt
                }
            text_to_prompt[text]['spans'].append((start_frame, end_frame))
        
        # Convert to the new format
        grouped_prompts = []
        for text, info in text_to_prompt.items():
            grouped_prompts.append((text, info['spans'], info['is_positive'], info['is_sequence_prompt']))
        
        return grouped_prompts

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
            max_frame_idx = lengths[i] - 1
            
            seq_prompts = self._get_prompts_from_annotations(sample.get("sequence_annotations", {}), max_frame_idx, True)
            frame_prompts = self._get_prompts_from_annotations(sample.get("frame_annotations", {}), max_frame_idx, False)

            if self.mode == PromptGenerationMode.BOTH:
                all_individual_prompts = seq_prompts + frame_prompts
            elif self.mode == PromptGenerationMode.SEQUENCE_ANNOTATIONS:
                all_individual_prompts = seq_prompts
            elif self.mode == PromptGenerationMode.FRAME_ANNOTATIONS:
                all_individual_prompts = frame_prompts
            elif self.mode == PromptGenerationMode.FRAME_ANNOTATIONS_WITH_FALLBACK_TO_SEQUENCE:
                all_individual_prompts = frame_prompts if frame_prompts else seq_prompts
            else:
                raise ValueError(f"Unsupported PromptGenerationMode: {self.mode}")

            # Group prompts by text to combine spans for the same prompt
            sample_prompts = self._group_prompts_by_text(all_individual_prompts)

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

class HML3DCollateFn:
    def __init__(
        self,
        fps: int = 20,
        padding_value: float = 0.0,
        max_texts: int = 8
    ):
        self.fps = fps
        self.padding_value = padding_value
        self.max_texts = max_texts
    
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
        dataset_names: list[str] = ["hml3d"] * len(batch)
        
        all_prompts = []
        for i, sample in enumerate(batch):
            sample_prompts = []
            max_frame_idx = lengths[i] - 1
            
            texts = sample.get("texts", [])
            start_t = sample.get("start_t", 0.0)
            end_t = sample.get("end_t", 0.0)

            start_frame = int(start_t * self.fps)
            end_frame = min(int(end_t * self.fps), max_frame_idx)

            selected_texts = random.sample(texts, min(len(texts), self.max_texts))

            for text in selected_texts:
                if text:
                    # Each text gets its own prompt with one span
                    prompt = (text, [(start_frame, end_frame)], True, True)
                    sample_prompts.append(prompt)
                    
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