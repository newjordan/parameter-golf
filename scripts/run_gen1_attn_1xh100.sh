#!/bin/bash
# Gen_1_aTTn: AttnRes (Kimi-K2 depth attention) on 1xH100
# Replaces U-Net skip connections with learned depth attention over layer cache.
set -euo pipefail

export USE_ATTNRES=1
export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=200
export TRAIN_LOG_EVERY=50

export NCCL_IB_DISABLE=1

torchrun --standalone --nproc_per_node=1 train_gpt.py
