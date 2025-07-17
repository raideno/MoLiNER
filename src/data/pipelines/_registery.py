from ._base import BasePipeline

from .hml3d import HML3DPipeline
from .babel import (
    BabelPipeline,
    BabelActionCategoryPipeline,
    BabelProcLabelPipeline,
    BabelRawLabelPipeline
)
from .locate import LocatePipeline
from .standardized_locate import WindowingStandardizedLocatePipeline, ChunkingStandardizedLocatePipeline
    
PIPELINE_REGISTRY: dict[type[BasePipeline], list[str]] = {
    LocatePipeline: ["20", "locate"],
    BabelPipeline: ["babel"],
    HML3DPipeline: ["hml3d"],
    BabelActionCategoryPipeline: ["babel-act-cat", "babel-act_cat"],
    BabelProcLabelPipeline: ["babel-proc-label", "babel-proc_label"],
    BabelRawLabelPipeline: ["babel-raw-label", "babel-raw_label"],
    WindowingStandardizedLocatePipeline: ["20-windowing-standardized", "windowing-standardized-locate", "windowing_standardized_locate"],
    ChunkingStandardizedLocatePipeline: ["20-chunking-standardized", "chunking-standardized-locate", "chunking_standardized_locate"],
}

def get_pipeline(name: str) -> "BasePipeline":
    for pipeline_class, names in PIPELINE_REGISTRY.items():
        if name in names:
            return pipeline_class()
    
    all_names = [name for names in PIPELINE_REGISTRY.values() for name in names]
    raise ValueError(f"Unknown pipeline: {name}. Available pipelines: {all_names}")