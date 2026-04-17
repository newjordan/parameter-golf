#!/bin/bash
# Post-2k 4-track parallel ablation sweep for VortexHelix v5_s3 champion.
# - 4 tracks, one per H100 (1xGPU each, sequential within a track).
# - 200 steps, val once at the end (VAL_LOSS_EVERY=200), train log every 25.
# - Each track explores ONE dimension; configs differ by single env override.
#
# Wall-time budget: 1xGPU step_avg ~1.26s baseline -> ~252s/run + ~25s val/setup.
#                   3-5 configs/track * ~280s ~= 14-23 min/track. Roughly balanced.
#
# Pre-req: 2k 4xH100 baseline must be done (all 4 GPUs free). If you run this
# before then it will OOM-fight with the running job. The script does NOT poll;
# launch by hand once `nvidia-smi` shows all 4 GPUs idle.

set -euo pipefail

export DATA_PATH="${DATA_PATH:-/workspace/Fartmagic/data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/workspace/Fartmagic/data/tokenizers/fineweb_1024_bpe.model}"

ITERATIONS=200
TRAIN_LOG_EVERY=25
VAL_LOSS_EVERY=200            # one val at step 0 (skipped) + one at end
MAX_WALLCLOCK_SECONDS=600     # hard cap per config; should never be hit

# v5_s3 champion kernel configs (commit 67ab894). Every track inherits these
# unless the per-config extra_env overrides one of them.
COMMON_KERNEL="VORTEX_BLOCK_SIZE=128 \
VORTEX_FWD_NUM_WARPS=8 VORTEX_FWD_NUM_STAGES=3 \
VORTEX_BWD_CHAOS_NUM_WARPS=8 VORTEX_BWD_CHAOS_NUM_STAGES=1 \
VORTEX_BWD_ATTN_NUM_WARPS=8 VORTEX_BWD_ATTN_NUM_STAGES=3 \
VORTEX_BWD_ATTN_DKV_NUM_WARPS=8 VORTEX_BWD_ATTN_DKV_NUM_STAGES=1"

# IMPORTANT: keep grad_clip OFF (matches the 1xGPU diagnostic baseline A and the
# 4xGPU 2k run uses 1.0 only as a NaN guard for DDP all-reduce; single-GPU here
# does NOT all-reduce, so leaving it at 0 keeps step time honest and matches A's
# 2.6472 anchor. Anything that toggles GRAD_CLIP changes both speed AND loss.

mkdir -p logs/post2k_ablations

# ----------------------------------------------------------------------------
# run_one: launch a single config in the background, wait for it.
#   $1 gpu, $2 track, $3 config_name, $4 extra_env (env-var string)
# ----------------------------------------------------------------------------
run_one () {
  local gpu=$1
  local track=$2
  local name=$3
  local extra_env=$4
  local logf="logs/post2k_ablations/${track}_${name}.log"
  echo "[GPU${gpu}][${track}] ${name} -> ${logf}"
  env CUDA_VISIBLE_DEVICES=${gpu} ${COMMON_KERNEL} ${extra_env} \
    ITERATIONS=${ITERATIONS} \
    TRAIN_LOG_EVERY=${TRAIN_LOG_EVERY} \
    VAL_LOSS_EVERY=${VAL_LOSS_EVERY} \
    MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS} \
    DATA_PATH=${DATA_PATH} \
    TOKENIZER_PATH=${TOKENIZER_PATH} \
    /venv/main/bin/python test_vortex_2k.py > "${logf}" 2>&1
}

# ----------------------------------------------------------------------------
# TRACK T1 (GPU0): chaos depth sweep — find the new convergence/cost knee
# 1xGPU diagnostic showed depth=5 (A) vs depth=0 (C) is +0.0042 BPB / 2.3x slower.
# We want the curve in between to see if depth=2 or 3 keeps most of the BPB win.
# ----------------------------------------------------------------------------
track_T1_chaos_depth () {
  run_one 0 T1 d0 "VORTEX_CHAOS_DEPTH=0"
  run_one 0 T1 d2 "VORTEX_CHAOS_DEPTH=2"
  run_one 0 T1 d3 "VORTEX_CHAOS_DEPTH=3"
  run_one 0 T1 d5 "VORTEX_CHAOS_DEPTH=5"   # champion anchor
  run_one 0 T1 d7 "VORTEX_CHAOS_DEPTH=7"
}

# ----------------------------------------------------------------------------
# TRACK T2 (GPU1): forward kernel autotune — warps x stages around (8,3) champion
# Pure speed signal (BPB should be deterministic for fwd config given same seed,
# but we still measure it as a regression guard). 5 configs.
# ----------------------------------------------------------------------------
track_T2_fwd_tune () {
  run_one 1 T2 w4_s3  "VORTEX_FWD_NUM_WARPS=4 VORTEX_FWD_NUM_STAGES=3"
  run_one 1 T2 w8_s2  "VORTEX_FWD_NUM_WARPS=8 VORTEX_FWD_NUM_STAGES=2"
  run_one 1 T2 w8_s3  "VORTEX_FWD_NUM_WARPS=8 VORTEX_FWD_NUM_STAGES=3"  # anchor
  run_one 1 T2 w8_s4  "VORTEX_FWD_NUM_WARPS=8 VORTEX_FWD_NUM_STAGES=4"
  run_one 1 T2 w16_s3 "VORTEX_FWD_NUM_WARPS=16 VORTEX_FWD_NUM_STAGES=3"
}

# ----------------------------------------------------------------------------
# TRACK T3 (GPU2): backward attn (dQ + dK/dV) tuning — biggest bwd cost
# Champion: ATTN w=8/s=3, ATTN_DKV w=8/s=1. Sweep 4 perturbations.
# Same BPB rationale as T2; main signal is step_avg.
# ----------------------------------------------------------------------------
track_T3_bwd_attn_tune () {
  run_one 2 T3 attn_w8s3_dkv_w8s1 \
    "VORTEX_BWD_ATTN_NUM_WARPS=8 VORTEX_BWD_ATTN_NUM_STAGES=3 \
     VORTEX_BWD_ATTN_DKV_NUM_WARPS=8 VORTEX_BWD_ATTN_DKV_NUM_STAGES=1"   # anchor
  run_one 2 T3 attn_w8s2_dkv_w8s1 \
    "VORTEX_BWD_ATTN_NUM_WARPS=8 VORTEX_BWD_ATTN_NUM_STAGES=2 \
     VORTEX_BWD_ATTN_DKV_NUM_WARPS=8 VORTEX_BWD_ATTN_DKV_NUM_STAGES=1"
  run_one 2 T3 attn_w8s3_dkv_w8s2 \
    "VORTEX_BWD_ATTN_NUM_WARPS=8 VORTEX_BWD_ATTN_NUM_STAGES=3 \
     VORTEX_BWD_ATTN_DKV_NUM_WARPS=8 VORTEX_BWD_ATTN_DKV_NUM_STAGES=2"
  run_one 2 T3 attn_w4s3_dkv_w4s1 \
    "VORTEX_BWD_ATTN_NUM_WARPS=4 VORTEX_BWD_ATTN_NUM_STAGES=3 \
     VORTEX_BWD_ATTN_DKV_NUM_WARPS=4 VORTEX_BWD_ATTN_DKV_NUM_STAGES=1"
  run_one 2 T3 attn_w8s4_dkv_w8s2 \
    "VORTEX_BWD_ATTN_NUM_WARPS=8 VORTEX_BWD_ATTN_NUM_STAGES=4 \
     VORTEX_BWD_ATTN_DKV_NUM_WARPS=8 VORTEX_BWD_ATTN_DKV_NUM_STAGES=2"
}

# ----------------------------------------------------------------------------
# TRACK T4 (GPU3): backward chaos warps + BLOCK_SIZE sweep
# We have NO data on BLOCK_SIZE=64 vs 128 since v5_s3 fixed it at 128. T=1024
# gives 8 q-blocks at BS=128, 16 at BS=64. Smaller blocks = more parallelism but
# more shared-mem pressure on chaos depth=5. Plus 3 chaos-warps perturbations.
# ----------------------------------------------------------------------------
track_T4_bs_and_chaos_warps () {
  run_one 3 T4 bs128_chaos_w8s1 ""                                            # anchor
  run_one 3 T4 bs64_chaos_w8s1  "VORTEX_BLOCK_SIZE=64"
  run_one 3 T4 bs128_chaos_w4s1 "VORTEX_BWD_CHAOS_NUM_WARPS=4"
  run_one 3 T4 bs128_chaos_w8s2 "VORTEX_BWD_CHAOS_NUM_STAGES=2"
  run_one 3 T4 bs128_chaos_w16s1 "VORTEX_BWD_CHAOS_NUM_WARPS=16"
}

# ----------------------------------------------------------------------------
# Launch all 4 tracks in parallel; each track is sequential internally.
# ----------------------------------------------------------------------------
echo "Launching 4 ablation tracks at $(date -u +%FT%TZ)"
track_T1_chaos_depth        &  T1_PID=$!
track_T2_fwd_tune           &  T2_PID=$!
track_T3_bwd_attn_tune      &  T3_PID=$!
track_T4_bs_and_chaos_warps &  T4_PID=$!

echo "T1 PID=${T1_PID}  T2 PID=${T2_PID}  T3 PID=${T3_PID}  T4 PID=${T4_PID}"
echo "Tail logs in logs/post2k_ablations/"

wait ${T1_PID} && echo "[T1] DONE" || echo "[T1] FAILED"
wait ${T2_PID} && echo "[T2] DONE" || echo "[T2] FAILED"
wait ${T3_PID} && echo "[T3] DONE" || echo "[T3] FAILED"
wait ${T4_PID} && echo "[T4] DONE" || echo "[T4] FAILED"

echo "All ablation tracks complete at $(date -u +%FT%TZ)"
echo "Summarize with: grep -E 'val_bpb|step_avg' logs/post2k_ablations/*.log"
