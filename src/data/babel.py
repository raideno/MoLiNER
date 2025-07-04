import os
import typing
import src.auth

try:
    import datasets
except ImportError:
    datasets = None

from src.data.pipelines import get_pipeline
from src.data.batching import SimplifiedBabelCollateFn

from src.constants import (
    DEFAULT_FPS,
    DEFAULT_PADDING_VALUE,
    HUGGING_FACE_TOKEN,
    BABEL_REMOTE_DATASET_NAME,
    MAP_AUGMENTATION_BATCH_SIZE,
    DEFAULT_LOAD_FROM_CACHE_FILE
)

class BabelDataset:
    """
    Wrapper class for Babel dataset with pipeline-based processing.
    """
    
    def __init__(
        self,
        split: str = "train",
        pipeline: str = "locate",
        load_from_cache_file: bool = DEFAULT_LOAD_FROM_CACHE_FILE,
    ):
        """
        Initialize Babel dataset with a processing pipeline.
        
        Args:
            split: Dataset split ("train", "validation", "test")
            pipeline: Name of the processing pipeline to use
            load_from_cache_file: Whether to load from cache file
        """
        if datasets is None:
            raise ImportError("datasets library is required but not available")
            
        self.split = split
        self.pipeline_name = pipeline
        self.load_from_cache_file = load_from_cache_file
        
        raw_dataset = datasets.load_dataset(
            BABEL_REMOTE_DATASET_NAME,
            trust_remote_code=True,
            name="full_all_motion"
        )
        
        self._pipeline = get_pipeline(pipeline)
        self._dataset = self._pipeline.apply(
            raw_dataset[split], 
            load_from_cache_file=load_from_cache_file
        )
        
        self._collate_fn = self._pipeline.get_collate_fn()
    
    @property
    def dataset(self):
        """Get the processed dataset for the specified split."""
        return self._dataset
    
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
    
