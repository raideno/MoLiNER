import os
import typing
import src.auth
import datasets

from hydra.utils import instantiate
from src.data.filtering import create_babel_filter_fn, FilterConfig

from src.data.typing import ProcessedBatch, RawBatch
from src.data.batching import PromptGenerationMode
from src.data.batching import BabelCollateFn, babel_create_raw_batch_collate_fn, babel_augment_and_split_batch

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
    Wrapper class for Babel dataset with collate function as a property.
    """
    
    def __init__(
        self,
        split: str = "train",
        split_prompts_types: bool = False,
        mode: PromptGenerationMode = PromptGenerationMode.BOTH,
        fps: int = 20,
        padding_value: float = 0.0,
        use_filtering: bool = False,
        filter_config: typing.Optional[typing.Union[dict, FilterConfig]] = None,
        load_from_cache_file: bool = DEFAULT_LOAD_FROM_CACHE_FILE,
    ):
        """
        Initialize Babel dataset.
        
        Args:
            split: Dataset split ("train", "validation", "test")
            split_prompts_types: Whether to split samples with both sequence and frame annotations
            mode: Prompt generation mode
            fps: Frames per second for time conversion
            padding_value: Value to use for padding
            use_filtering: Whether to apply filtering to the dataset
            filter_config: Filtering configuration (dict or FilterConfig instance)
            load_from_cache_file: Whether to load from cache file
        """
        if datasets is None:
            raise ImportError("datasets library is required but not available")
            
        self.split = split
        self.split_prompts_types = split_prompts_types
        self.mode = mode
        self.fps = fps
        self.padding_value = padding_value
        self.load_from_cache_file = load_from_cache_file
        
        self._dataset = datasets.load_dataset(
            BABEL_REMOTE_DATASET_NAME,
            trust_remote_code=True,
            name="full_all_motion"
        )
        
        if self.split_prompts_types:
            if self.split == "train":
                self._dataset["train"] = self._dataset["train"].map(
                    babel_augment_and_split_batch,
                    batched=True,
                    batch_size=MAP_AUGMENTATION_BATCH_SIZE,
                    load_from_cache_file=self.load_from_cache_file
                )
            elif self.split == "validation":
                self._dataset["validation"] = self._dataset["validation"].map(
                    babel_augment_and_split_batch,
                    batched=True,
                    batch_size=MAP_AUGMENTATION_BATCH_SIZE,
                    load_from_cache_file=self.load_from_cache_file
                )
        
        if use_filtering and self.split in ["train", "validation"]:
            if filter_config is None:
                raise ValueError("`filter_config` must be provided when `use_filtering` is True.")
            
            # If filter_config is already a FilterConfig instance, use it directly
            if isinstance(filter_config, FilterConfig):
                config = filter_config
                # Override fps if needed
                if config.fps != self.fps:
                    config = FilterConfig(
                        seed=config.seed,
                        fps=self.fps,
                        min_motion_frames=config.min_motion_frames,
                        max_motion_frames=config.max_motion_frames,
                        min_prompts_per_sample=config.min_prompts_per_sample,
                        max_prompts_per_sample=config.max_prompts_per_sample,
                        prompt_text_filter_fn=config.prompt_text_filter_fn,
                        min_span_frames=config.min_span_frames,
                        max_span_frames=config.max_span_frames,
                        max_spans_per_prompt=config.max_spans_per_prompt,
                        debug=config.debug
                    )
            else:
                # Handle dict-based config for backward compatibility
                filter_config_copy = dict(filter_config)
                
                # Handle the prompt_text_filter_fn separately if it's a config
                prompt_filter_conf = filter_config_copy.pop("prompt_text_filter_fn", None)
                
                if prompt_filter_conf:
                    prompt_text_filter_fn = instantiate(prompt_filter_conf)
                else:
                    prompt_text_filter_fn = None
                
                # Create FilterConfig instance
                config = FilterConfig(
                    **filter_config_copy,
                    prompt_text_filter_fn=prompt_text_filter_fn,
                    fps=self.fps
                )
            
            filter_fn = create_babel_filter_fn(config)
            
            self._dataset[self.split] = self._dataset[self.split].map(
                filter_fn,
                batched=True,
                batch_size=MAP_AUGMENTATION_BATCH_SIZE,
            )
        
        self._collate_fn = BabelCollateFn(
            fps=fps,
            mode=mode,
            padding_value=padding_value,
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
    
def get_babel_dataset_and_collate_fn(
    split_prompts_types: bool = False,
    mode: PromptGenerationMode = PromptGenerationMode.BOTH,
    load_from_cache_file: bool = DEFAULT_LOAD_FROM_CACHE_FILE
):
    """
    Legacy function for backward compatibility.
    Returns the full dataset and collate function.
    """
    if datasets is None:
        raise ImportError("datasets library is required but not available")
        
    babel_dataset = datasets.load_dataset(
        BABEL_REMOTE_DATASET_NAME,
        trust_remote_code=True,
        name="full_all_motion"
    )
    
    if split_prompts_types:
        babel_dataset["train"] = babel_dataset["train"].map(
            babel_augment_and_split_batch,
            batched=True,
            batch_size=MAP_AUGMENTATION_BATCH_SIZE,
            load_from_cache_file=load_from_cache_file
        )
        babel_dataset["validation"] = babel_dataset["validation"].map(
            babel_augment_and_split_batch,
            batched=True,
            batch_size=MAP_AUGMENTATION_BATCH_SIZE,
            load_from_cache_file=load_from_cache_file
        )

    babel_collate_fn = BabelCollateFn(
        fps=20,
        mode=mode,
        padding_value=0.0,
    )
    
    return babel_dataset, babel_collate_fn