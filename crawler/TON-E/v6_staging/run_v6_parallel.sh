#!/bin/bash
# v6 parallel noise-probe launcher.
# Fires 4 short (300-iter) 1xGPU runs CONCURRENTLY on GPUs 0/1/2/3,
# one per FIRST_LAYER_NOISE mode {off, flat, brazil, token_seeded}.
# Goal: compare early-training trajectories across noise modes in ~10 min.
#
# DO NOT RUN while GPUs are occupied by the v5 chain. Verify with
#   ps -ef | grep -E 'test_vortex|torchrun' | grep -v grep
# before launching this script.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

ts="$(date +%Y%m%d_%H%M%S)"

echo "[v6-parallel ${ts}] launching 4 short runs on GPUs 0,1,2,3"

CUDA_VISIBLE_DEVICES=0 bash run_v6_noise_off.sh    > logs/run_v6_off_${ts}.log    2>&1 &
pid_off=$!
CUDA_VISIBLE_DEVICES=1 bash run_v6_noise_flat.sh   > logs/run_v6_flat_${ts}.log   2>&1 &
pid_flat=$!
CUDA_VISIBLE_DEVICES=2 bash run_v6_noise_brazil.sh > logs/run_v6_brazil_${ts}.log 2>&1 &
pid_brazil=$!
CUDA_VISIBLE_DEVICES=3 bash run_v6_noise_seeded.sh > logs/run_v6_seeded_${ts}.log 2>&1 &
pid_seeded=$!

echo "[v6-parallel] pids: off=${pid_off} flat=${pid_flat} brazil=${pid_brazil} seeded=${pid_seeded}"
echo "[v6-parallel] logs: logs/run_v6_{off,flat,brazil,seeded}_${ts}.log"
echo "[v6-parallel] waiting for all 4..."

wait ${pid_off}    && echo "[v6-parallel] off    exited OK"    || echo "[v6-parallel] off    FAILED (rc=$?)"
wait ${pid_flat}   && echo "[v6-parallel] flat   exited OK"    || echo "[v6-parallel] flat   FAILED (rc=$?)"
wait ${pid_brazil} && echo "[v6-parallel] brazil exited OK"    || echo "[v6-parallel] brazil FAILED (rc=$?)"
wait ${pid_seeded} && echo "[v6-parallel] seeded exited OK"    || echo "[v6-parallel] seeded FAILED (rc=$?)"

echo
echo "[v6-parallel] ========== val_bpb summary =========="
for m in off flat brazil seeded; do
  log="logs/run_v6_${m}_${ts}.log"
  if [[ -f "${log}" ]]; then
    vb=$(grep -E 'val_bpb|val.*bpb' "${log}" | tail -1 || true)
    if [[ -n "${vb}" ]]; then
      echo "  ${m}: ${vb}"
    else
      echo "  ${m}: (no val_bpb line found in ${log})"
    fi
  else
    echo "  ${m}: (log missing: ${log})"
  fi
done
echo "[v6-parallel] ======================================"
