# Create Data Pipeline

It is possible to create your custom pipeline for processing motion-language datasets with custom filtering, augmentation, and batch structure transformations.

## Pipeline Architecture

The pipeline system is built around the `BasePipeline` class, which applies a sequence of processing steps to datasets. Each pipeline can include:

- **Batch Structure Simplification**: Convert complex dataset structures into standardized formats
- **Filtering**: Remove samples based on various criteria (motion length, prompt quality, sources, etc.)
- **Augmentation**: Apply transformations like span standardization, windowing, or chunking

## Available Pipelines

MoLiNER comes with several predefined pipelines:

| Pipeline Name                   | Description                                            | Use Case                        |
| ------------------------------- | ------------------------------------------------------ | ------------------------------- |
| `babel`                         | Basic Babel processing with simplified batch structure | General Babel dataset usage     |
| `babel-act-cat`                 | Babel with action category annotations only            | Action classification tasks     |
| `babel-proc-label`              | Babel with processed label annotations only            | Clean label experiments         |
| `babel-raw-label`               | Babel with raw label annotations only                  | Raw text processing             |
| `hml3d`                         | Basic HumanML3D processing                             | HumanML3D dataset usage         |
| `locate`                        | Babel with LOCATE class filtering (20 classes)         | Motion localization tasks       |
| `windowing-standardized-locate` | LOCATE with sliding window span standardization        | Fixed-length span experiments   |
| `chunking-standardized-locate`  | LOCATE with chunking span standardization              | Non-overlapping span processing |

## Creating a Custom Pipeline

### 1. Basic Pipeline Structure

Create a new pipeline by inheriting from `BasePipeline`:

`src/data/pipelines/my_custom.py`
```python
from src.data.pipelines._base import BasePipeline
from src.data.utils.batching import babel_simplify_batch_structure
from src.data.utils.filtering import (
    FilterConfig,
    FilterFunction
)
from src.data.utils.augmentation import (
    StandardizeSpansSlidingWindow,
)

class MyCustomPipeline(BasePipeline):
    def __init__(self):
        super().__init__("my-custom-pipeline")

        self.add_step(
            babel_simplify_batch_structure,
            batched=False
        )

        self.add_step(
            StandardizeSpansSlidingWindow(
                target_span_length=32,
                max_extend_frames=4
            ),
            batched=False
        )
        
        self.add_step(
            FilterFunction(FilterConfig(
                min_motion_frames=1,
                min_prompts_per_sample=1,
                sources=["proc_label"],
                min_span_frames=1,
                annotation_types=["frames", "sequence"]
            )),
            batched=True
        )
```

The steps will later be automatically executed before yout dataset is used.

**NOTE:** It is required to add the `babel_simplify_batch_structure` step as all of the predefined functions expect the data to be in the format returne by `babel_simplify_batch_structure`.

### 2. Registering Your Pipeline

To make your pipeline available throughout the system:

1. **Add to the registry** in `src/data/pipelines/_registery.py`:

```python
from .my_custom import MyCustomPipeline

PIPELINE_REGISTRY: dict[type[BasePipeline], list[str]] = {
    # ...existing pipelines...
    MyCustomPipeline: ["my-custom", "custom"],
}
```

2. **Import in the module** `src/data/pipelines/__init__.py`:

```python
from .my_custom import MyCustomPipeline
```

3. **Use in configurations**:

```yaml
# configs/data/my_dataset.yaml
_target_: src.data.babel.BabelDataset
pipeline: my-custom

motion_normalizer:
  _target_: src.helpers.motion_normalizer.MotionNormalizer
  stats_path: ./statistics/babel.pt
```

## Available Processing Primitives

### Filtering Options

The `FilterConfig` class provides comprehensive filtering capabilities:

```python
from src.data.utils.filtering import FilterConfig, FilterFunction

filter_config = FilterConfig(
    ```python
    # NOTE: motion length constraints
    min_motion_frames=1,            # Minimum number of motion frames
    max_motion_frames=4096,         # Maximum number of motion frames

    # NOTE: prompt constraints
    min_prompts_per_sample=1,       # Minimum prompts per sample
    max_prompts_per_sample=None,    # Maximum prompts per sample
    split_max_prompts_per_sample=False,             # Move exceding prompts to a copy sample

    # NOTE: span constraints
    min_span_frames=1,              # Minimum span length
    max_span_frames=None,           # Maximum span length

    # NOTE: prompt spans constraints
    min_spans_per_prompt=None,           # Maximum spans per prompt
    max_spans_per_prompt=None,           # Maximum spans per prompt           

    # NOTE: source filtering
    sources=["act_cat", "proc_label", "raw_label"],  # Allowed annotation sources

    # NOTE: annotation type filtering
    annotation_types=["frames", "sequence"],        # Allowed annotation types

    # NOTE: text filtering
    prompt_text_filter_function=None,               # Custom text filter function

    # NOTE: random seed
    seed=42                                         # Random seed for reproducible sampling

    debug=False                                     # Enable debug mode
)
```

### Pre-built Text Filters

```python
from src.data.utils.filtering import (
    create_locate_classes_filter_function,
    create_babel_20_classes_filter_function,
    create_babel_60_classes_filter_function,
    create_babel_90_classes_filter_function,
    create_babel_120_classes_filter_function,
    NoTransitionFilter,
    ExactMatchFilter
)

# Use predefined class filters
filter_config.prompt_text_filter_function = create_locate_classes_filter_function()

# Or create custom filters
filter_config.prompt_text_filter_function = NoTransitionFilter()  # Remove transition annotations
filter_config.prompt_text_filter_function = ExactMatchFilter(["walking", "running"])  # Only specific classes
```

### Span Standardization

Standardize span lengths using windowing or chunking approaches:

```python
from src.data.utils.augmentation import (
    StandardizeSpansSlidingWindow,
    StandardizeSpansChunking,
    SeparateFrameAndSequenceSpans
)

# Sliding window approach (overlapping spans)
window_function = StandardizeSpansSlidingWindow(
    target_span_length=16,     # Target length for all spans
    max_extend_frames=4,       # Maximum frames to extend short spans
    stride=8                   # Overlap between windows
)

# Chunking approach (non-overlapping spans)
chunk_function = StandardizeSpansChunking(
    target_span_length=16,     # Target length for all spans
    max_extend_frames=4        # Maximum frames to extend short spans
)

# Separate frame and sequence spans into distinct samples
separate_function = SeparateFrameAndSequenceSpans(
    debug=False                # Enable debug output
)

# Add to pipeline
self.add_step(window_function, batched=False)
self.add_step(separate_function, batched=True)
```

### Batch Structure Simplification

Convert dataset structures to standardized formats:

```python
from src.data.utils.batching import (
    babel_simplify_batch_structure,
    hml3d_simplify_batch_structure
)

# For Babel datasets
self.add_step(babel_simplify_batch_structure, batched=False)

# For HumanML3D datasets
self.add_step(hml3d_simplify_batch_structure, batched=True)

# For KIT-ML datasets
...
```

## Example Usage

```python
from src.data.babel import BabelDataset

dataset = BabelDataset(
    split="train",
    pipeline="my-custom",
    load_from_cache_file=True
)

for sample in dataset:
    print(f"Motion shape: {sample['motion']['new_joints'].shape}")
    print(f"Prompts: {len(sample['prompts'])}")
```

This pipeline system provides the flexibility to create sophisticated data processing workflows while maintaining consistency and performance across different datasets and experimental setups.