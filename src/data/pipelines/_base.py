import typing
import datasets
import multiprocessing

from src.data.utils.batching import (
    hml3d_simplify_batch_structure,
    babel_simplify_batch_structure,
)
from src.data.utils.collator import SimpleBatchStructureCollator
from src.constants import (
    MAP_AUGMENTATION_BATCH_SIZE,
    DEFAULT_LOAD_FROM_CACHE_FILE,
    DEFAULT_PROC_COUNT
)

class BasePipeline:
    def __init__(self, name: typing.Optional[str] = None):
        self.name = name
        self.steps = []
        self.proc_count = DEFAULT_PROC_COUNT or multiprocessing.cpu_count()
        
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
                load_from_cache_file=load_from_cache_file,
                num_proc=self.proc_count
            )
            assert isinstance(processed_dataset, datasets.Dataset), "Each step must return a datasets.Dataset instance"
        
        return processed_dataset
        
    def get_collate_function(self) -> typing.Callable:
        """
        Get the appropriate collate function for this pipeline.
        """
        return SimpleBatchStructureCollator()