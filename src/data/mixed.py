import typing
import datasets

from src.types import DatasetSample

from src.data.hml3d import HML3DDataset
from src.data.babel import BabelDataset

class MixedDataset:
    def __init__(
        self,
        split: str,
        hml3d_pipeline: str,
        babel_pipeline: str,
        load_from_cache_file: bool = True,
        motion_normalizer: typing.Optional[object] = None,
        interleave: bool = False,
        max_hml3d_samples: typing.Optional[int] = None,
        max_babel_samples: typing.Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        hml3d = HML3DDataset(
            split=split,
            pipeline=hml3d_pipeline,
            load_from_cache_file=load_from_cache_file,
            motion_normalizer=motion_normalizer,
        )
        babel = BabelDataset(
            split=split,
            pipeline=babel_pipeline,
            load_from_cache_file=load_from_cache_file,
            motion_normalizer=motion_normalizer,
        )
        
        def ensure_sid(example):
            example['sid'] = str(example.get('sid', ''))
            return example

        # babel._dataset = babel._dataset.map(ensure_sid)
        # hml3d._dataset = hml3d._dataset.map(ensure_sid)
        
        # NOTE: this is done because there is an issue with the sid type, it is different in Babel and HML3D datasets, they need to be consistent for the interleave / concatenate to work properly
        babel._dataset = babel._dataset.remove_columns("sid")
        hml3d._dataset = hml3d._dataset.remove_columns("sid")

        if shuffle:
            hml3d._dataset = hml3d._dataset.shuffle(seed=seed)
            babel._dataset = babel._dataset.shuffle(seed=seed)

        # TODO: if no shuffling is done, only the first elements will be selected and this is problematic i think, we should do some shuffling when selecting max samples as well
        if max_hml3d_samples is not None:
            hml3d._dataset = hml3d._dataset.select(range(min(max_hml3d_samples, len(hml3d))))
        if max_babel_samples is not None:
            babel._dataset = babel._dataset.select(range(min(max_babel_samples, len(babel))))

        if interleave:
            self.dataset = datasets.interleave_datasets(
                datasets=[hml3d._dataset, babel._dataset],
                probabilities=None,
                seed=None,
                # first_exhausted, all_exhausted
                stopping_strategy="all_exhausted",
            )
        else:
            self.dataset = datasets.concatenate_datasets(
                dsets=[hml3d._dataset, babel._dataset],
            )

    def __getitem__(self, idx) -> DatasetSample:
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @property
    def fingerprint(self):
        return f"mixed-{self.dataset._fingerprint}"
