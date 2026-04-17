#!/bin/bash
# v6 chain: runs v6a -> v6b -> v6c sequentially, each on all 4xH100.
# Does NOT touch run_v5_chain.sh. Intended to be launched AFTER v5 finishes.
set -u
cd /workspace/sota_crawler
mkdir -p logs
echo "[chain $(date -u +%FT%TZ)] starting v6a" | tee -a logs/v6_chain.log
bash run_v6a_noise_flat.sh > logs/run_v6a_noise_flat.log 2>&1
echo "[chain $(date -u +%FT%TZ)] v6a exit=$?, starting v6b" | tee -a logs/v6_chain.log
bash run_v6b_noise_brazil.sh > logs/run_v6b_noise_brazil.log 2>&1
echo "[chain $(date -u +%FT%TZ)] v6b exit=$?, starting v6c" | tee -a logs/v6_chain.log
bash run_v6c_noise_seeded.sh > logs/run_v6c_noise_seeded.log 2>&1
echo "[chain $(date -u +%FT%TZ)] v6c exit=$?, chain done" | tee -a logs/v6_chain.log
