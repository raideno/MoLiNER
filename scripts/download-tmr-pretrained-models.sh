#!/bin/bash

echo "[download-tmr]: Setting up TMR pretrained models..."

echo "[download-tmr]: Creating models directory..."
if test -d models; then
    echo "[download-tmr]: Models directory already exists."
else
    mkdir -p models
    echo "[download-tmr]: Models directory created."
fi

echo "[download-tmr]: Downloading pretrained models from Google Drive..."
python -m gdown "https://drive.google.com/uc?id=1n6kRb-d2gKsk8EXfFULFIpaUKYcnaYmm"

if test -f tmr_models.tgz; then
    echo "[download-tmr]: Download completed successfully."
else
    echo "[download-tmr]: Download failed. Please check your internet connection and try again."
    exit 1
fi

echo "[download-tmr]: Verifying file integrity with MD5 checksum..."
EXPECTED_MD5="7b6d8814f9c1ca972f62852ebb6c7a6f"
ACTUAL_MD5=$(md5sum tmr_models.tgz | cut -d' ' -f1)

if [ "$ACTUAL_MD5" = "$EXPECTED_MD5" ]; then
    echo "[download-tmr]: MD5 checksum verification passed."
else
    echo "[download-tmr]: MD5 checksum verification failed!"
    echo "[download-tmr]: Expected: $EXPECTED_MD5"
    echo "[download-tmr]: Actual: $ACTUAL_MD5"
    echo "[download-tmr]: Please rerun this script to download the file again."
    rm tmr_models.tgz
    exit 1
fi

echo "[download-tmr]: Extracting models..."
tar xfzv tmr_models.tgz

echo "[download-tmr]: Cleaning up temporary files..."
rm tmr_models.tgz

echo "[download-tmr]: TMR pretrained models setup complete!"