import typing
import logging

from src.constants import (
    DEFAULT_DEBUG,
    DEFAULT_STRIDE
)

logger = logging.getLogger(__name__)

class StandardizeSpansChunking:
    """
    Callable class to standardize spans using a chunking approach.
    
    Chunking approach:
    - Long spans are split into non-overlapping chunks of a target length.
    - Short spans are extended if possible, otherwise discarded.
    - Each chunk becomes a separate prompt with the same text.
    """
    def __init__(
        self,
        target_span_length: int,
        max_extend_frames: int,
        debug: bool = DEFAULT_DEBUG
    ):
        """
        Args:
            target_span_length: Target number of frames for each span.
            max_extend_frames: Maximum number of frames to extend short spans.
            debug: Whether to print debug information.
        """
        self.target_span_length = target_span_length
        self.max_extend_frames = max_extend_frames
        self.debug = debug

    def __call__(self, sample: dict) -> dict:
        new_sample = {key: value for key, value in sample.items()}
        
        motion_length = len(sample["motion"]["new_joints"])
        max_frame_idx = motion_length - 1
        
        prompts_list = sample.get("prompts", [])
        standardized_prompts = []
        
        original_prompt_count = len(prompts_list)
        extended_spans = 0
        discarded_spans = 0
        chunked_spans = 0
        
        for prompt_data in prompts_list:
            text = prompt_data.get("text", "")
            span = prompt_data.get("span", [])
            source = prompt_data.get("source", "")
            is_sequence = prompt_data.get("is_sequence", True)
            
            if not span or not text:
                continue
                
            start_frame, end_frame = span
            current_span_length = end_frame - start_frame + 1
            
            if current_span_length == self.target_span_length:
                standardized_prompts.append(prompt_data)
                continue
            
            if current_span_length < self.target_span_length:
                needed_frames = self.target_span_length - current_span_length
                
                if needed_frames <= self.max_extend_frames:
                    extend_each_side = needed_frames // 2
                    extend_remainder = needed_frames % 2
                    
                    new_start = max(0, start_frame - extend_each_side)
                    new_end = min(max_frame_idx, end_frame + extend_each_side + extend_remainder)
                    
                    achieved_length = new_end - new_start + 1
                    if achieved_length >= self.target_span_length:
                        if achieved_length > self.target_span_length:
                            trim_amount = achieved_length - self.target_span_length
                            new_end -= trim_amount
                        
                        extended_spans += 1
                        if self.debug:
                            logger.debug(f"[Chunking] Extended span from {current_span_length} to {new_end - new_start + 1} frames: '{text[:50]}...'")
                        
                        standardized_prompts.append({
                            "text": text,
                            "span": [new_start, new_end],
                            "source": source,
                            "is_sequence": is_sequence
                        })
                    else:
                        discarded_spans += 1
                        if self.debug:
                            logger.debug(f"[Chunking] Discarded span ({current_span_length} frames, needed {needed_frames}): '{text[:50]}...'")
                else:
                    discarded_spans += 1
                    if self.debug:
                        logger.debug(f"[Chunking] Discarded span ({current_span_length} frames, needed {needed_frames}): '{text[:50]}...'")
                continue
            
            if current_span_length > self.target_span_length:
                num_chunks = current_span_length // self.target_span_length
                chunked_spans += num_chunks
                
                if self.debug:
                    logger.debug(f"[Chunking] Splitting span of {current_span_length} frames into {num_chunks} chunks: '{text[:50]}...'")
                
                for chunk_idx in range(num_chunks):
                    chunk_start = start_frame + (chunk_idx * self.target_span_length)
                    chunk_end = chunk_start + self.target_span_length - 1
                    
                    chunk_end = min(chunk_end, end_frame, max_frame_idx)
                    
                    if chunk_end - chunk_start + 1 == self.target_span_length:
                        standardized_prompts.append({
                            "text": text,
                            "span": [chunk_start, chunk_end],
                            "source": source,
                            "is_sequence": is_sequence
                        })
                
                remainder_start = start_frame + (num_chunks * self.target_span_length)
                remainder_length = end_frame - remainder_start + 1
                
                if remainder_length > 0:
                    if self.debug:
                        logger.debug(f"[Chunking] Discarded remainder of {remainder_length} frames")
                    discarded_spans += 1
        
        new_sample["prompts"] = standardized_prompts
        
        if self.debug:
            final_prompt_count = len(standardized_prompts)
            logger.debug(f"[Chunking] Summary: {original_prompt_count} → {final_prompt_count} prompts "
                        f"(extended: {extended_spans}, chunked: {chunked_spans}, discarded: {discarded_spans})")
        
        return new_sample

class StandardizeSpansSlidingWindow:
    """
    Callable class to standardize spans using a sliding window approach.
    
    Sliding window approach:
    - Long spans are processed with overlapping windows of a target length.
    - Short spans are extended if possible, otherwise discarded.
    - Each window becomes a separate prompt with the same text.
    """
    def __init__(
        self,
        target_span_length: int,
        max_extend_frames: int,
        stride: int = DEFAULT_STRIDE,
        debug: bool = DEFAULT_DEBUG
    ):
        """
        Args:
            target_span_length: Target number of frames for each span.
            max_extend_frames: Maximum number of frames to extend short spans.
            stride: Step size for the sliding window.
            debug: Whether to print debug information.
        """
        self.target_span_length = target_span_length
        self.max_extend_frames = max_extend_frames
        self.stride = stride
        self.debug = debug

    def __call__(self, sample: dict) -> dict:
        new_sample = {key: value for key, value in sample.items()}
        
        motion_length = len(sample["motion"]["new_joints"])
        max_frame_idx = motion_length - 1
        
        prompts_list = sample.get("prompts", [])
        standardized_prompts = []
        
        original_prompt_count = len(prompts_list)
        extended_spans = 0
        discarded_spans = 0
        windowed_spans = 0
        
        for prompt_data in prompts_list:
            text = prompt_data.get("text", "")
            span = prompt_data.get("span", [])
            source = prompt_data.get("source", "")
            is_sequence = prompt_data.get("is_sequence", True)
            
            if not span or not text:
                continue
                
            start_frame, end_frame = span
            current_span_length = end_frame - start_frame + 1
            
            if current_span_length == self.target_span_length:
                standardized_prompts.append(prompt_data)
                continue
            
            if current_span_length < self.target_span_length:
                needed_frames = self.target_span_length - current_span_length
                
                if needed_frames <= self.max_extend_frames:
                    extend_each_side = needed_frames // 2
                    extend_remainder = needed_frames % 2
                    
                    new_start = max(0, start_frame - extend_each_side)
                    new_end = min(max_frame_idx, end_frame + extend_each_side + extend_remainder)
                    
                    achieved_length = new_end - new_start + 1
                    if achieved_length >= self.target_span_length:
                        if achieved_length > self.target_span_length:
                            trim_amount = achieved_length - self.target_span_length
                            new_end -= trim_amount
                        
                        extended_spans += 1
                        if self.debug:
                            logger.debug(f"[SlidingWindow] Extended span from {current_span_length} to {new_end - new_start + 1} frames: '{text[:50]}...'")
                        
                        standardized_prompts.append({
                            "text": text,
                            "span": [new_start, new_end],
                            "source": source,
                            "is_sequence": is_sequence
                        })
                    else:
                        discarded_spans += 1
                        if self.debug:
                            logger.debug(f"[SlidingWindow] Discarded span ({current_span_length} frames, needed {needed_frames}): '{text[:50]}...'")
                else:
                    discarded_spans += 1
                    if self.debug:
                        logger.debug(f"[SlidingWindow] Discarded span ({current_span_length} frames, needed {needed_frames}): '{text[:50]}...'")
                continue
            
            if current_span_length > self.target_span_length:
                window_positions = []
                window_start = start_frame
                
                while window_start + self.target_span_length - 1 <= end_frame:
                    window_end = window_start + self.target_span_length - 1
                    window_positions.append((window_start, window_end))
                    window_start += self.stride
                
                windowed_spans += len(window_positions)
                
                if self.debug:
                    logger.debug(f"[SlidingWindow] Creating {len(window_positions)} windows for span of {current_span_length} frames: '{text[:50]}...'")
                
                for window_start, window_end in window_positions:
                    window_end = min(window_end, max_frame_idx)
                    
                    if window_end - window_start + 1 == self.target_span_length:
                        standardized_prompts.append({
                            "text": text,
                            "span": [window_start, window_end],
                            "source": source,
                            "is_sequence": is_sequence
                        })
        
        new_sample["prompts"] = standardized_prompts
        
        if self.debug:
            final_prompt_count = len(standardized_prompts)
            logger.debug(f"[SlidingWindow] Summary: {original_prompt_count} → {final_prompt_count} prompts "
                        f"(extended: {extended_spans}, windowed: {windowed_spans}, discarded: {discarded_spans})")
        
        return new_sample

class SeparateFrameAndSequenceSpans:
    """
    Callable class to split samples with both sequence and frame annotations into two distinct samples.
    
    Split approach:
    - Samples containing both frame and sequence annotations are duplicated.
    - One copy contains only sequence annotations, the other only frame annotations.
    - Samples with only one type of annotation remain unchanged.
    """
    def __init__(self, debug: bool = DEFAULT_DEBUG):
        """
        Args:
            debug: Whether to print debug information.
        """
        self.debug = debug

    def __call__(self, batch: dict[str, list]) -> dict[str, list]:
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
                for key in batch.keys():
                    new_batch[key].append(batch[key][i])
        
        return new_batch