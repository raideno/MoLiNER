# WandB Integration for MoLiNER

This document explains how to use Weights & Biases (WandB) for experiment tracking in the MoLiNER project.

## Setup

**1. Login to WandB**:

```bash
wandb login
```

**2. Uncomment WandB code:**

Inside of [configs/trainer.yaml](../configs/trainer.yaml), uncomment the code corresponding to the wandb logger (`- ${loggers.wandb}`), and fill in the required values.

**3. You are all done:**

Now you can login to your WandB dashboard in order to monitor the trainings in real time.