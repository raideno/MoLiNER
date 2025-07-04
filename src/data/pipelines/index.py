import typing
import datasets

from src.data.batching import (
    SimplifiedBabelCollateFn,
    hml3d_simplify_batch_structure,
    babel_simplify_batch_structure,
)
from src.constants import (
    MAP_AUGMENTATION_BATCH_SIZE,
    DEFAULT_LOAD_FROM_CACHE_FILE
)

class BasePipeline:
    def __init__(self, name: str):
        self.name = name
        self.steps = []
        
    def add_step(self, step_fn: typing.Callable, batched: bool = False, batch_size: int = MAP_AUGMENTATION_BATCH_SIZE):
        """
        Add a processing step to the pipeline.
        """
        self.steps.append({
            'fn': step_fn,
            'batched': batched,
            'batch_size': batch_size
        })
        
    def apply(self, dataset: "datasets.Dataset", load_from_cache_file: bool = DEFAULT_LOAD_FROM_CACHE_FILE) -> "datasets.Dataset":
        """
        Apply all pipeline steps to the dataset.
        """
        processed_dataset = dataset
        
        for step in self.steps:
            processed_dataset = processed_dataset.map(
                step['fn'],
                batched=step['batched'],
                batch_size=step['batch_size'] if step['batched'] else None,
                load_from_cache_file=load_from_cache_file
            )
        
        return processed_dataset
        
    def get_collate_fn(self) -> typing.Callable:
        """
        Get the appropriate collate function for this pipeline.
        """
        return SimplifiedBabelCollateFn()

class BabelPipeline(BasePipeline):
    def __init__(self, name: typing.Optional[str] = None):
        super().__init__(name or "babel")
        
        self.add_step(babel_simplify_batch_structure, batched=False)

class HML3DPipeline(BasePipeline):
    """
    Pipeline for processing HML3D dataset with simplification.
    Applies simplification to convert 'texts' field to 'prompts' field.
    Uses SimplifiedBabelCollateFn since the structure becomes the same.
    """
    
    def __init__(self):
        super().__init__("hml3d")
        
        self.add_step(hml3d_simplify_batch_structure, batched=True)