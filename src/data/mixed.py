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
            # NOTE: alternate between datasets
            total = len(self)
            h_len = len(self.hml3d)
            b_len = len(self.babel)
            if index % 2 == 0:
                h_idx = index // 2
                if h_idx < h_len:
                    return self.hml3d[h_idx]
                else:
                    # NOTE: if HML3D exhausted, use Babel
                    b_idx = index - h_len
                    return self.babel[b_idx]
            else:
                b_idx = index // 2
                if b_idx < b_len:
                    return self.babel[b_idx]
                else:
                    h_idx = index - b_len
                    return self.hml3d[h_idx]
        else:
            # NOTE: concatenate datasets
            h_len = len(self.hml3d)
            if index < h_len:
                return self.hml3d[index]
            else:
                return self.babel[index - h_len]

    def __len__(self):
        return len(self.hml3d) + len(self.babel)

    @property
    def collate_function(self):
        return SimpleBatchStructureCollator()

    @property
    def fingerprint(self):
        return f"mixed-{self.hml3d.fingerprint}-{self.babel.fingerprint}"
