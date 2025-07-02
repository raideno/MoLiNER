import os
import typing
import src.auth
import datasets

from hydra.utils import instantiate

from src.data.batching import HML3DCollateFn, hml3d_create_raw_batch_collate_fn

from src.constants import (
    DEFAULT_FPS,
    DEFAULT_PADDING_VALUE,
    HUGGING_FACE_TOKEN,
    HML3D_REMOTE_DATASET_NAME,
    DEFAULT_LOAD_FROM_CACHE_FILE
)

class HML3DDataset:
    """
    Wrapper class for HML3D dataset with collate function as a property.
    """
    
    def __init__(
        self,
        split: str = "train",
        max_texts: int = 8,
        fps: int = DEFAULT_FPS,
        padding_value: float = DEFAULT_PADDING_VALUE,
        load_from_cache_file: bool = DEFAULT_LOAD_FROM_CACHE_FILE
    ):
        """
        Initialize HML3D dataset.
        
        Args:
            split: Dataset split ("train", "validation", "test")
            max_texts: Maximum number of texts to use per sample
            fps: Frames per second for time conversion
            padding_value: Value to use for padding
        """
        if datasets is None:
            raise ImportError("datasets library is required but not available")
            
        self.split = split
        self.max_texts = max_texts
        self.fps = fps
        self.padding_value = padding_value
        self.load_from_cache_file = load_from_cache_file
        
        self._dataset = datasets.load_dataset(
            HML3D_REMOTE_DATASET_NAME,
            trust_remote_code=True,
            name="full_all_motion"
        )
        
        self._collate_fn = HML3DCollateFn(
            fps=fps,
            padding_value=padding_value,
            max_texts=max_texts,
        )
    
    @property
    def dataset(self):
        """Get the dataset for the specified split."""
        return self._dataset[self.split]
    
    @property
    def collate_fn(self):
        """Get the collate function."""
        return self._collate_fn
    
    def __getitem__(self, index):
        """Get item from dataset."""
        return self.dataset[index]
    
    def __len__(self):
        """Get dataset length."""
        return len(self.dataset)


def get_hml3d_dataset_and_collate_fn(
    max_texts: int = 8,
):
    """
    Legacy function for backward compatibility.
    Returns the full dataset and collate function.
    """
    if datasets is None:
        raise ImportError("datasets library is required but not available")
        
    hml3d_dataset = datasets.load_dataset(
        HML3D_REMOTE_DATASET_NAME,
        trust_remote_code=True,
        name="full_all_motion"
    )
    
    hml3d_collate_fn = HML3DCollateFn(
        fps=20,
        padding_value=0.0,
        max_texts=max_texts,
    )
    
    return hml3d_dataset, hml3d_collate_fn