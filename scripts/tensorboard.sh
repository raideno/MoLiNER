#!/bin/bash

# USAGE: bash scripts/tensorboard.sh <RUN_DIR>

if [ -z "$1" ]; then
  echo "[usage]: $0 <run_dir>"
  exit 1
fi

run_dir="$1"

tensorboard --logdir="${run_dir}/logs/version_0"