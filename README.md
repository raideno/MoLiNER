# MoLiNER: Motion-Language-based Instance Segmentation and Retrieval

MoLiNER for motion analysis, language-based segmentation and retrieval.

## Installation

To get started with MoLiNER, you need to set up the environment and install the required dependencies.

1. **Clone the repository:**

```bash
git clone https://github.com/raideno/MoLiNER.git
cd MoLiNER
```

2. **Setup Python Environment:**

```bash
python -m venv .venv
# On macOS and Linux
source .venv/bin/activate
# On Windows
.venv\Scripts\activate
# NOTE: installing the dependencies
pip install -r requirements.txt
```

3. **Hugging Face Authentication (Required):**

Rename the [`.env.example`](./.env.example) file and rename it to `.env`; replace the `xxx` value with the appropriate values.

4. **Required Pretrained Weights:**

```bash
bash scripts/download-tmr-pretrained-models.sh
```

## Dataset Preparation

This project supports multiple datasets: **Babel**, **HumanML3D**, and **KIT-ML** (upcoming). All data downloading, preprocessing, etc is done automatically. The only required thing is to specify the `HUGGING_FACE_TOKEN` inside of the `.env` as instructed in the previous step, this is required as the raw data is hosted on hugging face in a private repository.

All data related code is available inside of [`src/data`](./src/data/).

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management. This allows for a flexible and composable way to configure experiments.

The main configuration files are located in the [`configs/`](./configs/) directory.

- [`defaults.yaml`](./configs/defaults.yaml): Contains global default settings.
- [`train-model.yaml`](./configs/train-model.yaml), [`test-model.yaml`](./configs/test-model.yaml), etc.: Main configuration files for different scripts.
- [`configs/model/moliner.yaml`](./configs/model/moliner.yaml): Configuration for the model architecture (e.g., encoders, decoders).
- [`configs/data/`](./configs/data/): Configuration for datasets.
- [`configs/trainer.yaml`](./configs/trainer.yaml): Configuration for the PyTorch Lightning trainer.

You can override any configuration setting from the command line. For example:

```bash
python train-model.py data=babel/20 model=moliner trainer.trainer.max_epochs=100
```

This command will train the `moliner` model on the `babel` dataset for 100 epochs.

## Training

To train a new model, use the `train-model.py` script. You need to specify the model and data configurations.

```bash
HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false python train-model.py model=<model_name> data=<dataset_name>
```

For example:

```bash
HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false python train-model.py model=moliner data=babel/20-standardized-windowing
```

All the available variants of the model can be found at [`configs/model`](./configs/model/) and the different data variants can be found at [`configs/data`](./configs/data/).

| **Model Variants** | **Description**                     |
| ------------------ | ----------------------------------- |
| `moliner`          | Default MoLiNER model architecture. |

```yaml
_target_: src.model.MoLiNER

lr: 1e-4

defaults:
  # tmr/scratch, tmr/pretrained, tmr/frozen
  - motion_frames_encoder: tmr/scratch
  # deberta/frozen, deberta/pretrained, tmr/scratch, tmr/pretrained, tmr/frozen, clip/frozen, clip/pretrained
  - prompts_tokens_encoder: deberta/pretrained

  #  windowed/16, windowed/8, static/1-16, static/8-32
  - spans_generator: windowed/16

  # mlp/deberta, mlp/tmr, mlp/clip
  - prompt_representation_layer: mlp/deberta
  # transformer, endpoints, query, lstm, convolution, mlp, pooling/min, pooling/mean, pooling/max
  - span_representation_layer: endpoints

  # product
  - scorer: product

  # greedy/flat, greedy/nested, greedy/overlap
  - decoder: greedy/overlap
```

You can override the different components of the model with the available ones to create your own variant of the model. This can be done at CLI level or by creating your own `.yaml` file in the [`./configs/model`](./configs/model/) and using it when calling the [train-model](#training) script.

```bash
python train-model.py \
    data=babel/20-standardized-windowing \
    model=moliner \
    prompts_tokens_encoder=clip
```

| **Data Variants**                 | **Description**                                 |
| --------------------------------- | ----------------------------------------------- |
| `babel/base`                      | Babel dataset for motion-language segmentation. |
| `babel/20`                        | Babel dataset with sequence-level annotations.  |
| `babel/20-standardized-chunking`  | Babel dataset with chunk-based annotations.     |
| `babel/20-standardized-windowing` | Babel dataset with window-based annotations.    |
| `hml3d/base`                      | HumanML3D dataset for 3D motion-language tasks. |
| `kitml/base`                      | KIT-ML dataset for motion-language retrieval.   |

### Hardware Configuration

By default, the trainer is configured to use CUDA acceleration. You can override the accelerator and device settings in [`configs/trainer.yaml`](./configs/trainer.yaml).

**Note:** Once training started, a directory inside the [`out`](./out) directory will be created, model weights, logs, etc will be stored there, this directory will be referred to as `run_dir` in the rest of the documentation.

## Weights Extraction

Before running evaluation or inference, you might want to extract the model weights from the PyTorch Lightning checkpoint for easier loading.

```bash
python extract.py run_dir=<path_to_run_dir>
```

This will save the model weights in the run directory.

## Evaluation

### Model Testing

To test a model on the test split of a dataset, use the `test-model.py` script. You need to provide the path on which the model have been "trained".

```bash
HYDRA_FULL_ERROR=1 python test-model.py run_dir=<path_to_run_dir>
```

### Retrieval Evaluation

To evaluate the model's performance on in-motion retrieval tasks:

```bash
HYDRA_FULL_ERROR=1 python evaluate-retrieval.py run_dir=<path_to_run_dir>
```

This supports HumanML3D, Babel sequence-level, and Babel frame-level datasets.

### Segmentation Evaluation

To evaluate the model on segmentation tasks with the Babel frame-level dataset:

```bash
HYDRA_FULL_ERROR=1 python evaluate-segmentation.py run_dir=<path_to_run_dir>
```

## Inference

To use a trained model for inference, you can use the `interface.py` script or load the model in your own scripts using the helper functions in `src/load.py`.

## Gradio Web Interface

A Gradio web interface is available for interactive model evaluation and visualization. To launch it, run:

```bash
HYDRA_FULL_ERROR=1 python interface.py run_dir=<path_to_run_dir>
```

You can then access the interface in your web browser.

## Pre-trained Models

Upcoming...
