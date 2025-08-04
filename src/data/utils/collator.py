import torch
import typing
import dataclasses
import collections

from src.types import RawBatch
from src.types import DatasetSample
from src.constants import DEFAULT_FPS, DEFAULT_PADDING_VALUE

if typing.TYPE_CHECKING:
    from src.models.moliner.modules import BasePromptsTokensEncoder

class SimpleBatchStructureCollator:
    def __init__(self, encoder: typing.Optional["BasePromptsTokensEncoder"] = None):
        self.fps = DEFAULT_FPS
        self.padding_value = DEFAULT_PADDING_VALUE
        self.encoder = encoder
    
    def __call__(self, batch: typing.List[DatasetSample]) -> "RawBatch":
        # self._validate_batch(batch)
        
        raw_motions = [torch.tensor(s["motion"]["new_joints"], dtype=torch.float32) for s in batch]
        transformed_motions = [torch.tensor(s["motion"]["new_joint_vecs"], dtype=torch.float32) for s in batch]
        lengths = [m.shape[0] for m in raw_motions]
        
        padded_raw_motion = torch.nn.utils.rnn.pad_sequence(raw_motions, batch_first=True, padding_value=self.padding_value)
        padded_transformed_motion = torch.nn.utils.rnn.pad_sequence(transformed_motions, batch_first=True, padding_value=self.padding_value)
        mask = self._create_mask(lengths, padded_raw_motion.shape[1])
        
        sids = [s.get("sid", -1) for s in batch]
        amass_paths = [s["amass_file_relative_path"] for s in batch]
        dataset_names = ["babel"] * len(batch)
        
        prompts_raw = [self._extract_prompts(s) for s in batch]
        prompt_data = self._process_prompts(prompts_raw, padded_raw_motion.device) if self.encoder else {}
        
        return RawBatch(
            sid=sids,
            dataset_name=dataset_names,
            amass_relative_path=amass_paths,
            raw_motion=padded_raw_motion,
            transformed_motion=padded_transformed_motion,
            motion_mask=mask,
            prompts=prompts_raw,
            **prompt_data
        )
    
    def _validate_batch(self, batch: list[dict]) -> None:
        for sample in batch:
            motion = sample.get("motion")
            if not isinstance(motion, dict):
                raise ValueError("Sample 'motion' field must be a dictionary. Use 'full_all_motion' config.")
            if not all(field in motion for field in ["new_joints", "new_joint_vecs"]):
                raise ValueError("Missing 'new_joints' or 'new_joint_vecs'. Use 'full_all_motion' config.")
    
    def _create_mask(self, lengths: list[int], max_len: int) -> torch.Tensor:
        mask = torch.zeros(len(lengths), max_len, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        return mask
    
    def _extract_prompts(self, sample: DatasetSample) -> typing.List[tuple]:
        class PromptAggregate(typing.TypedDict):
            spans: typing.List[typing.Tuple[int, int]]
            is_sequence: bool

        def _default_prompt_aggregate() -> PromptAggregate:
            return {"spans": [], "is_sequence": True}

        prompts_dict: collections.defaultdict[str, PromptAggregate] = collections.defaultdict(_default_prompt_aggregate)
        
        for prompt_data in sample["prompts"]:
            text = prompt_data["text"]
            span = prompt_data["span"]
            is_sequence = prompt_data["is_sequence"]
            
            if text and span:
                prompts_dict[text]["spans"].append(span)
                prompts_dict[text]["is_sequence"] = is_sequence
        
        return [(text, data["spans"], data["is_sequence"]) for text, data in prompts_dict.items()]
    
    def _process_prompts(self, prompts_raw: list, device: torch.device) -> dict:
        if not self.encoder:
            return self._empty_prompt_tensors(len(prompts_raw), device)
        
        batch_size = len(prompts_raw)
        max_prompts = max((len(p) for p in prompts_raw), default=0)
        
        if max_prompts == 0:
            return self._empty_prompt_tensors(batch_size, device)
        
        all_texts = [text for prompts in prompts_raw for text, _, _ in prompts]
        if not all_texts:
            return self._empty_prompt_tensors(batch_size, device)
            
        tokenized = self.encoder.tokenize(all_texts, padding=True, truncation=True, return_tensors='pt')
        max_seq_len = tokenized["input_ids"].shape[1]
        max_spans = max((len(spans) for prompts in prompts_raw for _, spans, _ in prompts), default=0)
        
        batch_results = []
        text_idx = 0
        
        for prompts in prompts_raw:
            if not prompts:
                batch_results.append(self._empty_batch_item(max_prompts, max_spans, max_seq_len))
                continue
                
            num_prompts = len(prompts)
            prompt_ids = tokenized["input_ids"][text_idx:text_idx + num_prompts]
            prompt_mask = tokenized["attention_mask"][text_idx:text_idx + num_prompts]
            text_idx += num_prompts
            
            spans_tensor = torch.zeros(num_prompts, max_spans, 2, dtype=torch.long)
            spans_mask = torch.zeros(num_prompts, max_spans, dtype=torch.bool)
            is_sequence = torch.zeros(num_prompts, dtype=torch.bool)
            
            for i, (_, spans, seq_flag) in enumerate(prompts):
                is_sequence[i] = seq_flag
                if spans:
                    spans_tensor[i, :len(spans)] = torch.tensor(spans)
                    spans_mask[i, :len(spans)] = True
            
            prompt_ids = torch.nn.functional.pad(prompt_ids, (0, 0, 0, max_prompts - num_prompts))
            prompt_mask = torch.nn.functional.pad(prompt_mask, (0, 0, 0, max_prompts - num_prompts))
            spans_tensor = torch.nn.functional.pad(spans_tensor, (0, 0, 0, 0, 0, max_prompts - num_prompts))
            spans_mask = torch.nn.functional.pad(spans_mask, (0, 0, 0, max_prompts - num_prompts))
            is_sequence = torch.nn.functional.pad(is_sequence, (0, max_prompts - num_prompts))
            
            prompt_exists = torch.zeros(max_prompts, dtype=torch.bool)
            prompt_exists[:num_prompts] = True
            
            batch_results.append({
                'prompt_ids': prompt_ids,
                'prompt_mask': prompt_mask,
                'spans_tensor': spans_tensor,
                'spans_per_prompt_mask': spans_mask,
                'spans_mask': prompt_exists,
                'is_sequence_tensor': is_sequence
            })
        
        return {
            'prompt_input_ids': torch.stack([r['prompt_ids'] for r in batch_results]).to(device),
            'prompt_attention_mask': torch.stack([r['prompt_mask'] for r in batch_results]).to(device),
            'target_spans': torch.stack([r['spans_tensor'] for r in batch_results]).to(device),
            'target_spans_mask': torch.stack([r['spans_mask'] for r in batch_results]).to(device),
            'target_spans_per_prompt_mask': torch.stack([r['spans_per_prompt_mask'] for r in batch_results]).to(device),
            'is_sequence_prompt': torch.stack([r['is_sequence_tensor'] for r in batch_results]).to(device)
        }
    
    def _empty_prompt_tensors(self, batch_size: int, device: torch.device) -> dict:
        return {
            'prompt_input_ids': torch.empty(batch_size, 0, 0, dtype=torch.long, device=device),
            'prompt_attention_mask': torch.empty(batch_size, 0, 0, dtype=torch.long, device=device),
            'target_spans': torch.empty(batch_size, 0, 0, 2, dtype=torch.long, device=device),
            'target_spans_mask': torch.empty(batch_size, 0, dtype=torch.bool, device=device),
            'target_spans_per_prompt_mask': torch.empty(batch_size, 0, 0, dtype=torch.bool, device=device),
            'is_sequence_prompt': torch.empty(batch_size, 0, dtype=torch.bool, device=device)
        }
    
    def _empty_batch_item(self, max_prompts: int, max_spans: int, max_seq_len: int) -> dict:
        return {
            'prompt_ids': torch.zeros(max_prompts, max_seq_len, dtype=torch.long),
            'prompt_mask': torch.zeros(max_prompts, max_seq_len, dtype=torch.long),
            'spans_tensor': torch.zeros(max_prompts, max_spans, 2, dtype=torch.long),
            'spans_mask': torch.zeros(max_prompts, dtype=torch.bool),
            'spans_per_prompt_mask': torch.zeros(max_prompts, max_spans, dtype=torch.bool),
            'is_sequence_tensor': torch.zeros(max_prompts, dtype=torch.bool)
        }
