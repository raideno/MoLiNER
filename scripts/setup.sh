#!/bin/bash

echo "[setup]: Setting up the MoLiNER environment..."
if test -f .venv/bin/activate; then
    echo "[setup]: Virtual environment already exists."
else
    echo "[setup]: Creating a new virtual environment..."
    python -m venv .venv
fi

echo "[setup]: Activating the virtual environment..."
if test -f .venv/bin/activate; then
    source .venv/bin/activate
else
    echo "[setup]: Virtual environment activation failed. Please check if the .venv directory have been created."
    exit 1
fi

echo "[setup]: Installing required packages..."
pip install -r requirements.txt

echo "[setup]: Downloading pretrained models..."
bash download-tmr-pretrained-models

echo "[setup]: Environment setup complete. Ready to use MoLiNER."