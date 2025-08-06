import torch
import typing

from src.types import Batch

def create_windows(
    window_size: int,
    stride: int,
    batch: Batch,
    use_raw_motion: bool
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sliding windows from motion data using fully vectorized operations.
    
    Args:
        window_size: Size of each window
        stride: Stride between windows
        batch: Input batch containing motion data
        use_raw_motion:
            - If True, use raw_motion (shape: batch_size, max_frames, 22, 3).
            - If False, use transformed_motion (shape: batch_size, max_frames, 263).
    
    Returns:
        windowed_motion: (total_windows, window_size, feature_dim) where feature_dim is (22, 3) for raw_motion or 263 for transformed_motion
        window_metadata: (total_windows, 3) -> [batch_idx, start_frame, end_frame]
        windows_per_sample: (batch_size,) -> number of windows per batch element
    """
    if use_raw_motion:
        # raw_motion shape: (batch_size, max_frames, 22, 3) -> (batch_size, max_frames, 66)
        motion_data = batch.raw_motion.reshape(batch.raw_motion.shape[0], batch.raw_motion.shape[1], -1)
    else:
        # transformed_motion shape: (batch_size, max_frames, 263)
        motion_data = batch.transformed_motion
    
    batch_size, max_frames, feature_dim = motion_data.shape
    device = motion_data.device
    
    # NOTE: (batch_size,)
    valid_lengths = batch.motion_mask.sum(dim=1)
    
    # NOTE: number of windows per sequence
    windows_per_sample = torch.clamp(
        ((valid_lengths - window_size) // stride) + 1,
        min=1
    )
    
    total_windows = int(windows_per_sample.sum().item())
    
    # NOTE: output tensors
    windowed_motion = torch.zeros(total_windows, window_size, feature_dim, device=device)
    window_metadata = torch.zeros(total_windows, 3, dtype=torch.long, device=device)
    
    window_start_idx = 0
    
    batch_indices = []
    start_positions = []
    
    for batch_idx in range(batch_size):
        valid_len = int(valid_lengths[batch_idx].item())
        num_windows = int(windows_per_sample[batch_idx].item())
        
        if valid_len < window_size:
            windowed_motion[window_start_idx, :valid_len] = motion_data[batch_idx, :valid_len]
            window_metadata[window_start_idx] = torch.tensor([batch_idx, 0, valid_len - 1], dtype=torch.long, device=device)
            window_start_idx += 1
        else:
            # NOTE: (valid_len, feature_dim)
            sequence = motion_data[batch_idx, :valid_len]
            
            # NOTE: (valid_len, feature_dim) -> (feature_dim, valid_len) -> (feature_dim, num_windows, window_size) -> (num_windows, window_size, feature_dim)
            windows = motion_data[batch_idx, :valid_len].transpose(0, 1).unfold(1, window_size, stride).permute(1, 2, 0)
            
            end_idx = window_start_idx + num_windows
            windowed_motion[window_start_idx:end_idx] = windows
            
            start_frames = torch.arange(num_windows, device=device) * stride
            batch_idx_tensor = torch.full((num_windows,), batch_idx, dtype=torch.long, device=device)
            end_frames = start_frames + window_size - 1
            
            window_metadata[window_start_idx:end_idx, 0] = batch_idx_tensor
            window_metadata[window_start_idx:end_idx, 1] = start_frames
            window_metadata[window_start_idx:end_idx, 2] = end_frames
            
            window_start_idx = end_idx
    
    if use_raw_motion:
        # (total_windows, window_size, 66) -> (total_windows, window_size, 22, 3)
        windowed_motion = windowed_motion.view(total_windows, window_size, 22, 3)
    
    return windowed_motion, window_metadata, windows_per_sample
