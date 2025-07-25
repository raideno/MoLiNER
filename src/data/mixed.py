import typing

from src.data.hml3d import HML3DDataset
from src.data.babel import BabelDataset

from src.data.utils.collator import SimpleBatchStructureCollator

class MixedDataset:
    """
    Dataset that mixes samples from HumanML3D and Babel datasets.
    """
    def __init__(
        self,
        split: str,
        hml3d_pipeline: str,
        babel_pipeline: str,
        load_from_cache_file: bool = True,
        motion_normalizer: typing.Optional[object] = None,
        interleave: bool = False,
    ):
        """
        Initialize MixedDataset with HML3D and Babel datasets.
        Args:
            split: Dataset split ("train", "validation", "test")
            hml3d_pipeline: Pipeline for HML3D
            babel_pipeline: Pipeline for Babel
            load_from_cache_file: Whether to load from cache
            motion_normalizer: Optional motion normalizer
            interleave: If True, alternate samples from each dataset
        """
        self.hml3d = HML3DDataset(
            split=split,
            pipeline=hml3d_pipeline,
            load_from_cache_file=load_from_cache_file,
            motion_normalizer=motion_normalizer,
        )
        self.babel = BabelDataset(
            split=split,
            pipeline=babel_pipeline,
            load_from_cache_file=load_from_cache_file,
            motion_normalizer=motion_normalizer,
        )
        self.interleave = interleave

    def __getitem__(self, index):
        if self.interleave:
            hml3d_len = len(self.hml3d)
            babel_len = len(self.babel)
            
            min_len = min(hml3d_len, babel_len)
            
            interleaved_len = min_len * 2
            
            if index < interleaved_len:
                if index % 2 == 0:
                    return self.hml3d[index // 2]
                else:
                    return self.babel[index // 2]
            else:
                # NOTE: a dataset is exhausted, continue with the other
                if hml3d_len > babel_len:
                    h_idx = index - babel_len
                    return self.hml3d[h_idx]
                else:
                    b_idx = index - hml3d_len
                    return self.babel[b_idx]
        else:
            # NOTE: concatenate datasets
            hml3d_len = len(self.hml3d)
            if index < hml3d_len:
                return self.hml3d[index]
            else:
                return self.babel[index - hml3d_len]

    def __len__(self):
        return len(self.hml3d) + len(self.babel)

    @property
    def collate_function(self):
        return SimpleBatchStructureCollator()

    @property
    def fingerprint(self):
        return f"mixed-{self.hml3d.fingerprint}-{self.babel.fingerprint}"
