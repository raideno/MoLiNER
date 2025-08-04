import torch
import typing
import random

from src.types import RawBatch
from src.models.moliner.modules import BasePromptsTokensEncoder

def create_random(
    batch_size: int = 2,
    max_frames_per_motion: int = 100,
    max_prompts_per_motion: int = 3,
    max_spans_per_prompt: int = 2,
    device: torch.device = torch.device('cpu'),
    encoder: typing.Optional["BasePromptsTokensEncoder"] = None
) -> "RawBatch":
    """
    Create a random RawBatch for testing purposes.
    
    Args:
        batch_size: Number of motions in the batch
        max_frames_per_motion: Maximum number of frames per motion
        max_prompts_per_motion: Maximum number of prompts per motion
        max_spans_per_prompt: Maximum number of spans per prompt
        device: Device to place tensors on
        encoder: Optional encoder for tokenizing prompts
        
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
    
    return RawBatch(
        sid=sid,
        dataset_name=dataset_name,
        amass_relative_path=amass_relative_path,
        raw_motion=raw_motion,
        transformed_motion=transformed_motion,
        motion_mask=motion_mask,
        prompts=prompts
    )