# MoLiNER (Motion LiNER)

## Todos (By Priority)

- [ ] Add tests to make sure 100% the loss is correctly working.
- [ ] Perform some statistics about the dataset and spans.
- [ ] Modify or create a variant of the static span generator to accept a lowerK and upperK rather than a single K, it'll then for each position it'll generate spans of size from lowerK to upperK. lower and upper K can be derived from the dataset it self by looking at the shortest and longest prompt.
- [ ] Add some post processing on the decoding step to clean up the predictions and merge any two spans into a longer one that should be merged.
- [ ] Implement a better segmentation and retrieval evaluation.
- [ ] Implement a pipeline training for TMR on HML3D and Babel.
- [ ] Missing span representation methods: cat (similar to query but simpler), mlp (brute force)

---

## Environment Setup

#### 1. Environment Configuration

Copy the [`.env.example`](./.env.example) file and rename it to `.env`. Replace the necessary fields with the correct values.

The `TOKEN` environment variable is required for downloading datasets from Hugging Face.

#### 2. Python Environment Setup

```bash
python -m venv .venv
# On macOS and Linux
source .venv/bin/activate
# On Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** All commands below assume your Python environment is properly set up and activated.

## Testing and Development

<!-- #### 1. Implementation Tests

Test the implementations of different model components to ensure they work properly. This is useful when modifying any submodule of the model.

```bash
pytest -v -s
``` -->

#### 2. Model Testing

Test dataset loading, dataloader creation, and forward pass through the model. You can use [`pdb`](https://docs.python.org/3/library/pdb.html) for debugging if enabled in the configuration.

```bash
HYDRA_FULL_ERROR=1 python test.py
```

## Configuration

This project uses [Hydra](https://hydra.cc/docs/intro/) for configuration management. All configurations are stored in the [`/configs`](/configs/) directory and can be modified as needed.

## Training

### Model Training

Train the model using the following command:

```bash
HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false python train-model.py \
    model=moliner \
    data=locate-babel
```

The training system supports multiple model architectures and dataset variants. You can combine any model variant with any compatible dataset variant by specifying them in the command above.

#### Available Model Variants

- **`moliner`** - Base MoLiNER model.
- **`moliner_pretrained-tmr`** - MoLiNER with pre-trained TMR encoder (non frozen).
- **`moliner_standardized-spans`** - MoLiNER with standardized sizes (16 frames).
- **`moliner_standardized-spans_pretrained-tmr`** - MoLiNER with both standardized (16 frames) spans and pre-trained TMR encoder (non frozen).

#### Available Dataset Variants

- **`babel`** - Standard Babel dataset.
- **`hml3d`** - HumanML3D dataset.
- **`locate-babel`** - Babel dataset with locate classes only.
- **`standardized-chunking-locate-babel`** - Babel with locate classes & standardized span size (chunks of 16 frames).
- **`standardized-windowing-locate-babel`** - Babel with locate classes & standardized span size (sliding windows of 16 frames).

#### Training Output

By default, training outputs (checkpoints, logs, etc.) are saved to timestamped directories under `out/` (e.g., `out/2025-07-07_14-30-15/`). This directory path is refered to as `RUN_DIR`in the subsequent commands.

### Weight Extraction

After training completes, extract the final model weights:

```bash
HYDRA_FULL_ERROR=1 python extract.py run_dir=RUN_DIR
```

**Parameters:**

- `RUN_DIR`: Path to the training output directory. This corresponds to the timestamped directory created during training, located in the `out/` directory by default.

## Evaluation

### Model Testing

Test the model on the test split:

```bash
HYDRA_FULL_ERROR=1 python test-model.py
```

### Retrieval Evaluation

Evaluate the model for in-motion retrieval. Supports HumanML3D dataset, Babel sequence-level dataset, and Babel frame-level dataset:

```bash
HYDRA_FULL_ERROR=1 python evaluate-retrieval.py
```

### Segmentation Evaluation

Evaluate the model for segmentation tasks. Available for Babel frame-level dataset:

```bash
HYDRA_FULL_ERROR=1 python evaluate-segmentation.py
```

## Gradio Web Interface

Launch an interactive web interface for real-time model evaluation and visualization:

```bash
HYDRA_FULL_ERROR=1 python interface.py
```
