#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RESULT_ROOT="${REPO_ROOT}/results/shroud_loopviz_smoke_$(date +%Y%m%d_%H%M%S)"
DATA_PATH_SMOKE="${DATA_PATH_SMOKE:-/tmp/shroud_loopviz_smoke_data}"
DATA_PATH_SOURCE="${DATA_PATH_SOURCE:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
SEED="${SEED:-1337}"
TAG="${TAG:-SHROUD_LOOPVIZ_BASE}"

mkdir -p "${RESULT_ROOT}"

export DATA_PATH_SMOKE DATA_PATH_SOURCE
python3 - <<'PY'
import os
from pathlib import Path
import numpy as np

src_root = Path(os.environ["DATA_PATH_SOURCE"])
out_root = Path(os.environ["DATA_PATH_SMOKE"])
out_root.mkdir(parents=True, exist_ok=True)

train_out = out_root / "fineweb_train_000000.bin"
val_out = out_root / "fineweb_val_000000.bin"
if train_out.exists() and val_out.exists():
    print(f"smoke dataset exists: {out_root}")
    raise SystemExit(0)

train_src = sorted(src_root.glob("fineweb_train_*.bin"))
val_src = sorted(src_root.glob("fineweb_val_*.bin"))
if not train_src or not val_src:
    raise FileNotFoundError(f"missing source shards under {src_root}")

header_words = 256
header_bytes = header_words * 4

def read_tokens(path: Path, n: int):
    header = np.fromfile(path, dtype="<i4", count=header_words)
    if header.size != header_words or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise RuntimeError(f"bad shard header: {path}")
    total = int(header[2])
    n = min(n, total)
    return np.fromfile(path, dtype="<u2", count=n, offset=header_bytes)

def write_shard(path: Path, tokens):
    header = np.zeros((header_words,), dtype="<i4")
    header[0] = 20240520
    header[1] = 1
    header[2] = int(tokens.size)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype("<u2", copy=False).tobytes())

write_shard(train_out, read_tokens(train_src[0], 65536))
write_shard(val_out, read_tokens(val_src[0], 32768))
print(f"created smoke dataset: {out_root}")
PY

RUN_ID="${TAG}_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${RESULT_ROOT}/${TAG}.log"
TRACE_JSONL="${RESULT_ROOT}/${TAG}.trace.jsonl"
TRACE_POINTS="${RESULT_ROOT}/${TAG}.trace.points.json"

echo "=== SHROUD LOOPVIZ SMOKE ==="
echo "result_root=${RESULT_ROOT}"
echo "run_id=${RUN_ID}"
echo "trace_jsonl=${TRACE_JSONL}"
echo "trace_points=${TRACE_POINTS}"
echo "============================"

set +e
env \
  SEED="${SEED}" \
  RUN_ID="${RUN_ID}" \
  DATA_PATH="${DATA_PATH_SMOKE}" \
  MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-45}" \
  ITERATIONS="${ITERATIONS:-24}" \
  WARMUP_STEPS=0 \
  VAL_LOSS_EVERY=0 \
  TRAIN_LOG_EVERY=4 \
  TRAIN_BATCH_TOKENS="${TRAIN_BATCH_TOKENS:-8192}" \
  TRAIN_SEQ_LEN="${TRAIN_SEQ_LEN:-256}" \
  EVAL_SEQ_LEN="${EVAL_SEQ_LEN:-256}" \
  VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16384}" \
  MODEL_DIM="${MODEL_DIM:-256}" \
  NUM_HEADS="${NUM_HEADS:-4}" \
  NUM_KV_HEADS="${NUM_KV_HEADS:-2}" \
  USE_CRAWLER=1 \
  NUM_FLAT_LAYERS=4 \
  NUM_CRAWLER_LAYERS=1 \
  CRAWLER_LOOPS="${CRAWLER_LOOPS:-4}" \
  CRAWLER_MLP_MULT="${CRAWLER_MLP_MULT:-4.0}" \
  INST_DIM="${INST_DIM:-32}" \
  DELTA_NET_HEADS=0 \
  CRAWLER_QUANT_INT8=1 \
  NGRAM_EVAL_ORDER=0 \
  SKIP_EMA=1 \
  SKIP_GPTQ=1 \
  LOOP_AWARE_GPTQ=0 \
  COMPILE_ENABLED=0 \
  COMPILE_FULLGRAPH=0 \
  TORCHDYNAMO_OPTIMIZE_DDP=0 \
  SHROUD_ENABLE=1 \
  SHROUD_STEP_EVERY=1 \
  SHROUD_HEAD_TRACE=1 \
  SHROUD_HEAD_MAX_TOKENS="${SHROUD_HEAD_MAX_TOKENS:-64}" \
  SHROUD_MAX_EVENTS="${SHROUD_MAX_EVENTS:-500000}" \
  SHROUD_TRACE_PATH="${TRACE_JSONL}" \
  python3 "${REPO_ROOT}/experiments/Shroud/train_gpt.py" 2>&1 | tee "${LOG_FILE}"
rc=$?
set -e

if [[ ! -f "${TRACE_JSONL}" ]]; then
  echo "ERROR: trace file missing: ${TRACE_JSONL}"
  exit 2
fi

python3 "${REPO_ROOT}/experiments/Shroud/visualizer/build_shroud_points.py" \
  --input "${TRACE_JSONL}" \
  --output "${TRACE_POINTS}" \
  --max-points "${MAX_POINTS:-180000}" \
  --max-edges "${MAX_EDGES:-180000}"

echo "run_status=$rc"
echo "log=${LOG_FILE}"
echo "trace_jsonl=${TRACE_JSONL}"
echo "trace_points=${TRACE_POINTS}"
echo "viewer=${REPO_ROOT}/experiments/Shroud/visualizer/shroud_viewer.html"

exit "${rc}"
