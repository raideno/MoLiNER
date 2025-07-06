from .index import BasePipeline, BabelPipeline, HML3DPipeline
from .locate import LocatePipeline
from .standardized_locate import WindowingStandardizedLocatePipeline, ChunkingStandardizedLocatePipeline
    
PIPELINE_REGISTRY = {
    "locate": LocatePipeline,
    "babel": BabelPipeline,
    "hml3d": HML3DPipeline,
    "windowing-standardized-locate": WindowingStandardizedLocatePipeline,
    "chunking-standardized-locate": ChunkingStandardizedLocatePipeline,
    "windowing_standardized_locate": WindowingStandardizedLocatePipeline,
    "chunking_standardized_locate": ChunkingStandardizedLocatePipeline,
}

def get_pipeline(name: str) -> BasePipeline:
    if name not in PIPELINE_REGISTRY:
        raise ValueError(f"Unknown pipeline: {name}. Available pipelines: {list(PIPELINE_REGISTRY.keys())}")
    
    return PIPELINE_REGISTRY[name]()