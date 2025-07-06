import enum
import torch
import typing
import random
import string
import dataclasses

# NOTE: this is to avoid circular import issues
if typing.TYPE_CHECKING:
    from src.model.modules.tokenizers.index import BaseTokenizer

@dataclasses.dataclass
class RawBatch:
    sid: typing.List[int]
    dataset_name: typing.List[str]
    amass_relative_path: typing.List[str]
    
    # NOTE: (batch_size, batch_max_frames_per_motion, 22, 3)
    raw_motion: torch.Tensor
    # NOTE: (batch_size, batch_max_frames_per_motion, 263)
    transformed_motion: torch.Tensor
    # NOTE: (batch_size, batch_max_frames_per_motion)
    # For each motion, set to 1 if the motion is valid, 0 otherwise (padding frame).
    motion_mask: torch.Tensor
    
    # NOTE: a list of lists, each inner list contains the prompts for the corresponding motion
    # A prompt is now a tuple of (text, list_of_spans, is_sequence_prompt)
    # where list_of_spans is a list of (start_frame, end_frame) tuples for this prompt
    # The is_sequence_prompt flag indicates if the prompt is a sequence-level prompt (True) or a frame-level prompt (False).
    prompts: typing.List[typing.List[typing.Tuple[str, typing.List[typing.Tuple[int, int]], bool]]]
    
    @staticmethod
    def _validate_type_or_raise(
        value: typing.Any,
        name: str = "value"
    ):
        if not isinstance(value, RawBatch):
            raise TypeError(f"Expected {name} to be of type {RawBatch}, but got {type(value)}.")
    
    def validate_type_or_raise(
        self,
        name: typing.Any = "value",
    ):
        RawBatch._validate_type_or_raise(self, name=name)
    
    @classmethod
    def create_random(
        cls,
        batch_size: int = 2,
        max_frames_per_motion: int = 100,
        max_prompts_per_motion: int = 3,
        max_spans_per_prompt: int = 2,
        device: torch.device = torch.device('cpu')
    ) -> "RawBatch":
        """
        Create a random RawBatch for testing purposes.
        
        Args:
            batch_size: Number of motions in the batch
            max_frames_per_motion: Maximum number of frames per motion
            max_prompts_per_motion: Maximum number of prompts per motion
            max_spans_per_prompt: Maximum number of spans per prompt
            device: Device to place tensors on
            
        Returns:
            A randomly generated RawBatch
        """
        sid = [random.randint(1000, 9999) for _ in range(batch_size)]
        dataset_name = [random.choice(['babel', 'hml3d', 'amass']) for _ in range(batch_size)]
        amass_relative_path = [f"path/to/motion_{i}.npz" for i in range(batch_size)]
        
        raw_motion = torch.randn(batch_size, max_frames_per_motion, 22, 3, device=device)
        transformed_motion = torch.randn(batch_size, max_frames_per_motion, 263, device=device)
        
        motion_mask = torch.zeros(batch_size, max_frames_per_motion, device=device)
        for i in range(batch_size):
            valid_frames = random.randint(50, max_frames_per_motion)
            motion_mask[i, :valid_frames] = 1
        
        action_words = ['walk', 'run', 'jump', 'sit', 'stand', 'dance', 'turn', 'bend', 'reach', 'grab']
        direction_words = ['forward', 'backward', 'left', 'right', 'up', 'down']
        
        prompts = []
        for i in range(batch_size):
            motion_prompts = []
            num_prompts = random.randint(1, max_prompts_per_motion)
            
            for j in range(num_prompts):
                action = random.choice(action_words)
                direction = random.choice(direction_words)
                text = f"person {action}s {direction}"
                
                num_spans = random.randint(1, max_spans_per_prompt)
                spans = []
                valid_frames = int(motion_mask[i].sum().item())
                
                for k in range(num_spans):
                    start_frame = random.randint(0, max(0, valid_frames - 20))
                    end_frame = random.randint(start_frame + 5, min(start_frame + 30, valid_frames))
                    spans.append((start_frame, end_frame))
                
                is_sequence_prompt = random.choice([True, False])
                
                motion_prompts.append((text, spans, is_sequence_prompt))
            
            prompts.append(motion_prompts)
        
        return cls(
            sid=sid,
            dataset_name=dataset_name,
            amass_relative_path=amass_relative_path,
            raw_motion=raw_motion,
            transformed_motion=transformed_motion,
            motion_mask=motion_mask,
            prompts=prompts
        )
        
# TODO: we could have multiple prompts of the same type in a single motion, so when encoding them we should only encode one of them and not all of them.
@dataclasses.dataclass
class ProcessedBatch:
    sid: typing.List[int]
    dataset_name: typing.List[str]
    amass_relative_path: typing.List[str]
    
    # --- Motion Data ---
    
    # NOTE: (batch_size, batch_max_frames_per_motion, motion_feature_dim)
    motion_features: torch.Tensor

    # NOTE: (batch_size, batch_max_frames_per_motion)
    motion_mask: torch.Tensor

    # --- Prompt Data (The "Labels" to be localized) ---
    
    # NOTE: (batch_size, batch_max_prompts_per_motion, batch_max_prompt_length_in_tokens)
    # The tokenized and padded text prompts.
    prompt_input_ids: torch.Tensor

    # NOTE: (batch_size, batch_max_prompts_per_motion, batch_max_prompt_length_in_tokens)
    # The attention mask for the tokenized prompts. A 1 indicates a real token, 0 indicates padding.
    prompt_attention_mask: torch.Tensor
    
    # --- Ground Truth Data (The Localizations of each prompt) ---
    
    # NOTE:: (batch_size, batch_max_prompts_per_motion, batch_max_spans_per_prompt, 2)
    # The last dim is [start_frame, end_frame]
    # Only required for training
    target_spans: typing.Optional[torch.Tensor] = None

    # NOTE: (batch_size, batch_max_prompts_per_motion)
    # A mask to indicate which prompts are real vs. padding, as each motion in the batch can have a different number of prompts.
    # Only required for training
    target_spans_mask: typing.Optional[torch.Tensor] = None
    
    # NOTE: (batch_size, batch_max_prompts_per_motion, batch_max_spans_per_prompt)
    # A mask to indicate which spans within each prompt are real vs. padding
    # Only required for training
    target_spans_per_prompt_mask: typing.Optional[torch.Tensor] = None
    
    # NOTE: (batch_size, batch_max_prompts_per_motion)
    # A boolean tensor indicating if the prompt is a sequence-level prompt (True) or a frame-level prompt (False).
    # Only required for training
    is_sequence_prompt: typing.Optional[torch.Tensor] = None
    
    def is_training_data_defined(self) -> bool:
        """
        Check if the batch contains training data (i.e., target spans and masks).
        
        Returns:
            bool: True if training data is defined, False otherwise.
        """
        return (
            self.target_spans is not None and
            self.target_spans_mask is not None and
            self.target_spans_per_prompt_mask is not None and
            self.is_sequence_prompt is not None
        )
    
    @staticmethod
    def _validate_type_or_raise(
        value: typing.Any,
        name: str = "value"
    ):
        if not isinstance(value, ProcessedBatch):
            raise TypeError(f"Expected {name} to be of type {ProcessedBatch}, but got {type(value)}.")
    
    def validate_type_or_raise(
        self,
        name: typing.Any = "value",
    ):
        ProcessedBatch._validate_type_or_raise(self, name=name)
    
    @classmethod
    def from_raw_batch(
        cls,
        raw_batch: RawBatch,
        tokenizer: "BaseTokenizer",
    ) -> "ProcessedBatch":
        batch_size = raw_batch.transformed_motion.shape[0]
        device = raw_batch.transformed_motion.device
        
        batch_max_prompts = max(len(prompts) for prompts in raw_batch.prompts)
        
        batch_max_spans_per_prompt = 0
        for batch_prompts in raw_batch.prompts:
            for prompt in batch_prompts:
                batch_max_spans_per_prompt = max(batch_max_spans_per_prompt, len(prompt[1]))
        
        # NOTE: first collect all texts to determine global max sequence length
        all_texts_in_batch = []
        for batch_prompts in raw_batch.prompts:
            for prompt in batch_prompts:
                all_texts_in_batch.append(prompt[0])
        
        if all_texts_in_batch:
            global_tokenized = tokenizer.tokenize(
                all_texts_in_batch,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            batch_max_sequence_length = global_tokenized["input_ids"].shape[1]
        else:
            batch_max_sequence_length = 1
        
        all_prompt_ids, all_prompt_masks = [], []
        all_target_spans = []
        all_target_spans_masks = []
        all_target_spans_per_prompt_masks = []
        all_is_sequence_prompt_flags = []

        for batch_index in range(batch_size):
            batch_motion_prompts = raw_batch.prompts[batch_index]
            
            texts = [prompt[0] for prompt in batch_motion_prompts]
            spans_lists = [prompt[1] for prompt in batch_motion_prompts]
            is_sequence_prompt_list = [prompt[2] for prompt in batch_motion_prompts]
            
            # NOTE: we tokenize the prompt texts
            if texts:
                tokenized = tokenizer.tokenize(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                prompt_ids = tokenized["input_ids"]
                prompt_mask = tokenized["attention_mask"]
                
                # Pad to global max sequence length if needed
                current_seq_len = prompt_ids.shape[1]
                if current_seq_len < batch_max_sequence_length:
                    pad_length = batch_max_sequence_length - current_seq_len
                    pad_ids = torch.full((prompt_ids.shape[0], pad_length), tokenizer.pad_token_id, dtype=torch.long)
                    prompt_ids = torch.cat([prompt_ids, pad_ids], dim=1)
                    pad_mask = torch.zeros((prompt_mask.shape[0], pad_length), dtype=torch.long)
                    prompt_mask = torch.cat([prompt_mask, pad_mask], dim=1)
            else:
                # NOTE: empty texts case
                prompt_ids = torch.zeros((0, batch_max_sequence_length), dtype=torch.long)
                prompt_mask = torch.zeros((0, batch_max_sequence_length), dtype=torch.long)

            number_of_prompts = len(batch_motion_prompts)
            number_of_paddings = batch_max_prompts - number_of_prompts

            # NOTE: pad prompt tensors
            pad_ids = torch.zeros((number_of_paddings, prompt_ids.shape[1]), dtype=torch.long)
            prompt_ids = torch.cat([prompt_ids, pad_ids], dim=0)
            pad_mask = torch.zeros((number_of_paddings, prompt_mask.shape[1]), dtype=torch.long)
            prompt_mask = torch.cat([prompt_mask, pad_mask], dim=0)

            # NOTE: create target spans tensor for this batch
            # Shape: (num_prompts, max_spans_per_prompt, 2)
            prompt_spans_tensor = torch.zeros((number_of_prompts, batch_max_spans_per_prompt, 2), dtype=torch.long)
            spans_per_prompt_mask = torch.zeros((number_of_prompts, batch_max_spans_per_prompt), dtype=torch.bool)
            
            for prompt_idx, spans_list in enumerate(spans_lists):
                num_spans = len(spans_list)
                if num_spans > 0:
                    spans_per_prompt_mask[prompt_idx, :num_spans] = True
                    for span_idx, (start_frame, end_frame) in enumerate(spans_list):
                        prompt_spans_tensor[prompt_idx, span_idx, 0] = start_frame
                        prompt_spans_tensor[prompt_idx, span_idx, 1] = end_frame

            # NOTE: pad target spans for padded prompts
            pad_spans = torch.zeros((number_of_paddings, batch_max_spans_per_prompt, 2), dtype=torch.long)
            spans_tensor = torch.cat([prompt_spans_tensor, pad_spans], dim=0)
            
            # NOTE: pad spans per prompt mask for padded prompts  
            pad_spans_per_prompt_mask = torch.zeros((number_of_paddings, batch_max_spans_per_prompt), dtype=torch.bool)
            spans_per_prompt_mask = torch.cat([spans_per_prompt_mask, pad_spans_per_prompt_mask], dim=0)
            
            # NOTE: pad the `is_sequence_prompt` tensor
            is_sequence_prompt_tensor = torch.tensor(is_sequence_prompt_list, dtype=torch.bool)
            # NOTE: we set padded prompts to False (not positive) but it does not matter as they are not used in training, ignored
            pad_is_sequence = torch.zeros(number_of_paddings, dtype=torch.bool)
            is_sequence_prompt_tensor = torch.cat([is_sequence_prompt_tensor, pad_is_sequence], dim=0)

            # NOTE: create the mask for all non-padded prompts (both positive and negative)
            spans_mask = torch.zeros(batch_max_prompts, dtype=torch.bool)
            spans_mask[:number_of_prompts] = True

            all_prompt_ids.append(prompt_ids)
            all_prompt_masks.append(prompt_mask)
            all_target_spans.append(spans_tensor)
            all_target_spans_masks.append(spans_mask)
            all_target_spans_per_prompt_masks.append(spans_per_prompt_mask)
            all_is_sequence_prompt_flags.append(is_sequence_prompt_tensor)
            
        return cls(
            sid=raw_batch.sid,
            dataset_name=raw_batch.dataset_name,
            amass_relative_path=raw_batch.amass_relative_path,
            motion_features=raw_batch.transformed_motion,
            motion_mask=raw_batch.motion_mask,
            prompt_input_ids=torch.stack(all_prompt_ids).to(device),
            prompt_attention_mask=torch.stack(all_prompt_masks).to(device),
            target_spans=torch.stack(all_target_spans).to(device),
            target_spans_mask=torch.stack(all_target_spans_masks).to(device),
            target_spans_per_prompt_mask=torch.stack(all_target_spans_per_prompt_masks).to(device),
            is_sequence_prompt=torch.stack(all_is_sequence_prompt_flags).to(device)
        )
        
@dataclasses.dataclass
class ForwardOutput:
    # The core output matrix from the pair scorer.
    # NOTE: (batch_size, batch_max_prompts_per_motion, batch_max_spans_per_motion_in_batch)
    similarity_matrix: torch.Tensor

    # The start and end frames for each candidate span generated by the model.
    # NOTE: (batch_size, batch_max_spans_per_motion_in_batch, 2) -> [start_frame, end_frame]
    candidate_spans_indices: torch.Tensor
    
    # A mask to identify which candidate spans are valid versus padding.
    # NOTE: (batch_size, batch_max_spans_per_motion_in_batch)
    candidate_spans_mask: torch.Tensor
    
    # --- --- --- xxx --- --- ---
    
    prompts_representation: torch.Tensor
    spans_representation: torch.Tensor
    prompts_mask: torch.Tensor
            
    # --- --- --- xxx --- --- ---
    
    @staticmethod
    def _validate_type_or_raise(
        value: typing.Any,
        name: str = "value"
    ):
        if not isinstance(value, ForwardOutput):
            raise TypeError(f"Expected {name} to be of type {ForwardOutput}, but got {type(value)}.")
    
    def validate_type_or_raise(
        self,
        name: typing.Any = "value",
    ):
        ForwardOutput._validate_type_or_raise(self, name=name)
        
class DecodingStrategy(enum.Enum):
    """
    Defines the strategy for handling overlapping spans during decoding.

    - FLAT: No overlaps are allowed. The highest-scoring non-overlapping spans are chosen.
    - NESTED: Allows spans that are fully nested within another selected span, but prohibits partial overlaps.
    - OVERLAP: Allows any overlap. All spans above the score threshold are selected.
    """
    FLAT = "flat"
    NESTED = "nested"
    OVERLAP = "overlap"

@dataclasses.dataclass
class EvaluationResult:
    """
    Holds the results of a single motion evaluation.
    """
    motion_length: int
    # NOTE: list[(prompt_text, start_frame, end_frame, score)]
    predictions: typing.List[typing.Tuple[str, int, int, float]]