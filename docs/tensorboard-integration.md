
# Tensorboard Integration for MoLiNER

This document explains how to use Tensorboard for experiment tracking in the MoLiNER project.

## Setup

**1. Uncomment Tensorboard code:**

Inside of [configs/trainer.yaml](../configs/trainer.yaml), uncomment the code corresponding to the wandb logger (`- ${loggers.tensorboard}`), and fill in the required values.

**2. Start Tensorboard:**

Run the [scripts/tensorboard.sh](../scripts/tensorboard.sh) script.

**3. Visit the Tensorboard Dashboard:**

You can now visit the dashboard using the address printed in the terminal.