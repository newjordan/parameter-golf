#!/bin/bash
# Poller helper: waits for any running `test_vortex_2k.py` processes to exit,
# then fires run_v6_chain.sh. Safe to run alongside the live v5 chain — it
# only observes, never signals. Launch manually with:
#   nohup bash /workspace/sota_crawler/launch_v6_after_v5.sh > /workspace/sota_crawler/logs/launch_v6_after_v5.log 2>&1 &
set -u
cd /workspace/sota_crawler
mkdir -p logs
LOG=/workspace/sota_crawler/logs/launch_v6_after_v5.log

echo "[poller $(date -u +%FT%TZ)] starting; will wait for test_vortex_2k.py to exit" | tee -a "$LOG"

# Wait loop: poll every 60s. We check for any test_vortex_2k.py python process
# (works for both the torchrun launcher and its workers).
while pgrep -f "test_vortex_2k.py" > /dev/null 2>&1; do
    sleep 60
done

echo "[poller $(date -u +%FT%TZ)] no test_vortex_2k.py processes detected; pausing 120s for GPU cleanup" | tee -a "$LOG"
sleep 120

# Double-check the GPUs actually went quiet before firing v6.
if pgrep -f "test_vortex_2k.py" > /dev/null 2>&1; then
    echo "[poller $(date -u +%FT%TZ)] test_vortex_2k.py came back; aborting v6 launch" | tee -a "$LOG"
    exit 1
fi

echo "[poller $(date -u +%FT%TZ)] launching run_v6_chain.sh" | tee -a "$LOG"
bash /workspace/sota_crawler/run_v6_chain.sh >> "$LOG" 2>&1
echo "[poller $(date -u +%FT%TZ)] v6 chain exit=$?" | tee -a "$LOG"
