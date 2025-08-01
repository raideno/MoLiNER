from ._base import BasePipeline

from ._registery import get_pipeline, PIPELINE_REGISTRY

from .babel import (
    BabelPipeline,
    BabelActionCategoryPipeline,
    BabelProcLabelPipeline,
    BabelRawLabelPipeline
)
from .hml3d import (
    HML3DPipeline,
    Max1024HML3DGroupedPipeline,
    Max1024HML3DSplittedPipeline,
    MlpMax1024HML3DSplittedPipeline,
    MlpMax1024HML3DGroupedPipeline,
)
from .locate import (
    LocatePipeline,
    WindowingStandardizedLocatePipeline,
    ChunkingStandardizedLocatePipeline,
    FilteredLocatePipeline
)