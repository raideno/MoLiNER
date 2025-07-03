# MoLiNER (Motion LiNER)

## Todos (By Priority)

- [ ] Add a hugging face interface using Gradio.
- [ ] Perform some statistics about the dataset and spans.
- [ ] Implement and understand decoding protocols.
- [ ] Modify or create a variant of the static span generator to accept a lowerK and upperK rather than a single K, it'll then for each position it'll generate spans of size from lowerK to upperK. lower and upper K can be derived from the dataset it self by looking at the shortest and longest prompt.
- [ ] Add some post processing on the decoding step to clean up the predictions and stick any two spans that should be.
- [ ] Add possibility to use pretrained TMR motion encoder and text encoder.
- [ ] Implement a better segmentation and retrieval evaluation.
- [ ] Implement a pipeline training for TMR on HML3D and Babel.
- [ ] Make the model save an example of retrieval every epoch on validation to have a look at the evolution of the predictions.
- [ ] Use einsum for matrix computations.
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
HYDRA_FULL_ERROR=1 python train-model.py
```

### Weight Extraction

Extract the model weights after training:

```bash
HYDRA_FULL_ERROR=1 python extract.py
```

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