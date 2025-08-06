#!/bin/bash

SESSION="segmentation-training"

MODELS=(
    # "segmentation.frozen-tmr.local.yaml"
    "segmentation.mlp.local.yaml"
    "segmentation.pretrained-tmr.local.yaml"
    # "segmentation.scratch-tmr.local.yaml"
    # "segmentation.stgcn.local.yaml"
)

DATA="babel/20/base"
ACCELERATOR="cuda"
DEVICES="[1]"

tmux new-session -d -s "$SESSION"

tmux rename-window -t "$SESSION:0" "nvtop"
tmux send-keys -t "$SESSION:0" "nvtop" Enter

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    WINDOW_NAME="model-$((i+1))-$(echo "$MODEL" | cut -d'.' -f2)"
    
    tmux new-window -t "$SESSION" -n "$WINDOW_NAME"
    WINDOW_TARGET="$SESSION:$WINDOW_NAME"
    
    tmux send-keys -t "$WINDOW_TARGET" "HYDRA_FULL_ERROR=1 python train-model.py data=$DATA model=$MODEL trainer.accelerator=$ACCELERATOR +trainer.devices=$DEVICES" Enter
done

echo "Tmux session '$SESSION' created with ${#MODELS[@]} windows for model training"
echo "To attach to the session: tmux attach-session -t $SESSION"
echo "To list windows: tmux list-windows -t $SESSION"
echo "To switch between windows: Ctrl+b then window number (0-$((${#MODELS[@]}-1)))"