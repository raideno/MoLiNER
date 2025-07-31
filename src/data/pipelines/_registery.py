from ._base import BasePipeline

from .hml3d import (
    HML3DPipeline,
    Max1024HML3DGroupedPipeline,
    Max1024HML3DSplittedPipeline,
    MlpMax1024HML3DSplittedPipeline,
    MlpMax1024HML3DGroupedPipeline,
)
from .babel import (
    BabelPipeline,
    BabelActionCategoryPipeline,
    BabelProcLabelPipeline,
    BabelRawLabelPipeline,
    BabelSeparate
)
from .locate import (
    LocatePipeline,
    ChunkingStandardizedLocatePipeline,
    WindowingStandardizedLocatePipeline,
    FilteredLocatePipeline
)
    
PIPELINE_REGISTRY: dict[type[BasePipeline], list[str]] = {
    BabelSeparate: ["babel-separate", "babel-separate"],
    FilteredLocatePipeline: ["filtered-locate", "filtered_locate"],
    LocatePipeline: ["20", "locate"],
    BabelPipeline: ["babel"],
    HML3DPipeline: ["hml3d"],
    BabelActionCategoryPipeline: ["babel-act-cat", "babel-act_cat"],
    BabelProcLabelPipeline: ["babel-proc-label", "babel-proc_label"],
    BabelRawLabelPipeline: ["babel-raw-label", "babel-raw_label"],
    WindowingStandardizedLocatePipeline: ["20-windowing-standardized", "windowing-standardized-locate", "windowing_standardized_locate"],
    ChunkingStandardizedLocatePipeline: ["20-chunking-standardized", "chunking-standardized-locate", "chunking_standardized_locate"],
    Max1024HML3DGroupedPipeline: ["max-1024-hml3d-grouped", "max_1024_hml3d_grouped"],
    Max1024HML3DSplittedPipeline: ["max-1024-hml3d-splitted", "max_1024_hml3d_splitted"],
    MlpMax1024HML3DGroupedPipeline: ["mlp-max-1024-hml3d-grouped", "mlp_max_1024_hml3d_grouped"],
    MlpMax1024HML3DSplittedPipeline: ["mlp-max-1024-hml3d-splitted", "mlp_max_1024_hml3d_splitted"],
}

def get_pipeline(name: str) -> "BasePipeline":
    for pipeline_class, names in PIPELINE_REGISTRY.items():
        if name in names:
            return pipeline_class()
    
    all_names = [name for names in PIPELINE_REGISTRY.values() for name in names]
    raise ValueError(f"Unknown pipeline: {name}. Available pipelines: {all_names}")