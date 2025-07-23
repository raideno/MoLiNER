import typing
import datasets

import src.auth

from src.data.pipelines import get_pipeline

from src.constants import (
    BABEL_REMOTE_DATASET_NAME,
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
        motion_normalizer: typing.Optional[object] = None,
    ):
        """
        Initialize Babel dataset with a processing pipeline.
        
        Args:
            split: Dataset split ("train", "validation", "test")
            pipeline: Name of the processing pipeline to use
            load_from_cache_file: Whether to load from cache file
            motion_normalizer: Optional motion normalizer object
        """
        if datasets is None:
            raise ImportError("datasets library is required but not available")
            
        self.split = split
        self.pipeline_name = pipeline
        self.load_from_cache_file = load_from_cache_file
        self.motion_normalizer = motion_normalizer
        
        raw_dataset = datasets.load_dataset(
            BABEL_REMOTE_DATASET_NAME,
            trust_remote_code=True,
            name="full_all_motion"
        )
        assert isinstance(raw_dataset, datasets.DatasetDict)
        
        self._pipeline = get_pipeline(self.pipeline_name)
        self._dataset = self._pipeline.apply(
            raw_dataset[split], 
            load_from_cache_file=load_from_cache_file
        )
        
        self._collate_function = self._pipeline.get_collate_function()

    @property
    def dataset(self):
        return self._dataset
    
    @property
    def collate_function(self):
        return self._collate_function
    
    @property
    def fingerprint(self):
        return self._dataset._fingerprint
    
    def __getitem__(self, index):
        item = self.dataset[index]
        if self.motion_normalizer and isinstance(item, dict):
            motion = item.get("motion", None)
            if motion is not None:
                normed = self.motion_normalizer.normalize(motion)
                item = item.copy()
                item["motion"] = normed
        return item
    
    def __len__(self):
        return len(self.dataset)

