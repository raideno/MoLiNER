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

Rename the [`.env.example`](./.env.example) to `.env` and replace the `xxx` value with the appropriate values.

4. **TMR Pretrained Weights:**

```bash
bash scripts/download-tmr-pretrained-models.sh
```

## Dataset Preparation

This project supports multiple datasets: **Babel**, **HumanML3D**, and **KIT-ML** (upcoming). All data downloading, preprocessing, etc is done automatically. The only required thing is to specify the `HUGGING_FACE_TOKEN` inside of the `.env` as instructed in the previous step, this is required as the raw data is hosted on hugging face in a private repository.

All data related code is available inside of [`src/data`](./src/data/) and the associated configuration inside of [`configs/data`](./configs/data/).

A set of predefined data pipelines with filtering are already available and listed in [**# Data Pipelines**](#data-pipelines). It is also possible to create your custom data pipeline with your custom filtering, instructions are available at [docs/create-data-pipeline.md](./docs/create-data-pipeline.md).

## Training

To train a new model, use the `train-model.py` script.

1. **Create the Model:**

Duplicate the [`configs/model/moliner.yaml`](./configs/model/moliner.yaml) file and **name it as you wish**.

2. **Specify the Modules:**

Replace all the `???` in the `.yaml` file with one of the possible values for each module.

3. **Start the Training:**

```bash
HYDRA_FULL_ERROR=1 TOKENIZERS_PARALLELISM=false python train-model.py model=<MODEL_NAME> data=<DATASET_NAME> trainer.accelerator=cuda
```

**NOTEs:**

- For more control on the trainer, you can change the [configs/trainer.yaml](./configs/trainer.yaml).
- `<MODEL_NAME>` should be set the the name of the file you just created without the `.yaml` extension.
- `<DATASET_NAME>` possible values can be found at [`configs/data`](./configs/data/) and are also listed below.

### Data Pipelines

| **Data Variants**                                                                              | **Description**                                                  |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [`babel/base`](./configs/data/babel/base.yaml)                                                 | Babel dataset for motion-language segmentation.                  |
| [`babel/separate`](./configs/data/babel/separate.yaml)                                         | Frame and sequence annotations are put in different samples.     |
| [`babel/20/base`](./configs/data/babel/20/base.yaml)                                           | Babel dataset with sequence-level annotations.                   |
| [`babel/20/standardized/chunking/16`](./configs/data/babel/20/standardized/chunking/16.yaml)   | Babel dataset with chunk-based annotations. 16 Frames per span.  |
| [`babel/20/standardized/windowing/16`](./configs/data/babel/20/standardized/windowing/16.yaml) | Babel dataset with window-based annotations. 16 Frames per span. |
| [`hml3d/base`](./configs/data/hml3d/base.yaml)                                                 | HumanML3D dataset for 3D motion-language tasks.                  |
| [`mixed/base`](./configs/data/mixed/base.yaml)                                                 | A mix of HML3D and Babel dataset.                                |

**`RUN_DIR`:** Once training started, a directory inside the [`out`](./out) directory will be created, model weights, logs, etc will be stored there, this directory will be referred to as `run_dir` in the rest of the documentation.

## Configurations

This project uses [Hydra](https://hydra.cc/) for configuration management. This allows for a flexible and composable way to configure experiments.

The main configuration files are located in the [`configs/`](./configs/) directory.

- [`defaults.yaml`](./configs/defaults.yaml): Contains global default settings.
- [`train-model.yaml`](./configs/train-model.yaml), [`test-model.yaml`](./configs/test-model.yaml), etc.: Main configuration files for different scripts.
- [`configs/model/moliner.yaml`](./configs/model/moliner.yaml): Configuration for the model architecture (e.g., encoders, decoders).
- [`configs/data/`](./configs/data/): Configuration for datasets.
- [`configs/trainer.yaml`](./configs/trainer.yaml): Configuration for the PyTorch Lightning trainer.

You can override any configuration setting from the command line. For example:

```bash
python train-model.py data=<data-name> model=<model-name> trainer.max_epochs=100
```

This command will train the specified model on the specified dataset for 100 epochs.

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
