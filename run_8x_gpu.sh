#!/bin/bash
# VortexHelix 8x GPU Training Launcher
# This script uses torchrun to launch the training across 8 GPUs.
# The code automatically scales the grad_accum_steps (8 // world_size).

# Adjust these paths to point to your Fineweb 10B dataset and Tokenizer on the new pod
export DATA_PATH="/workspace/.cache/huggingface/hub/datasets--willdepueoai--parameter-golf/snapshots/a85b0e6035c3c94bc23685a07c81a8f3bf89db80/datasets/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="/workspace/.cache/huggingface/hub/datasets--willdepueoai--parameter-golf/snapshots/a85b0e6035c3c94bc23685a07c81a8f3bf89db80/datasets/tokenizers/fineweb_1024_bpe.model"

# Hyperparameters for the run
export ITERATIONS=20000 # Scaling up to 20k steps for the full run!
export MAX_WALLCLOCK_SECONDS=14400 # 4 hours max wallclock

echo "🚀 Launching VortexHelix on 8 GPUs..."
torchrun --standalone --nproc_per_node=8 test_vortex_2k.py
