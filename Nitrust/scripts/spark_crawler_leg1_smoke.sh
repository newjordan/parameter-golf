#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RESULT_ROOT="${REPO_ROOT}/results/crawler_leg1_smoke_$(date +%Y%m%d_%H%M%S)"
DATA_PATH_SMOKE="${DATA_PATH_SMOKE:-/tmp/nitrust_smoke_data}"
DATA_PATH_SOURCE="${DATA_PATH_SOURCE:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
SO_PATH="${SO_PATH:-${REPO_ROOT}/Nitrust/rust/target/release/libnitrust_py.so}"
SEED="${SEED:-1337}"

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

write_shard(train_out, read_tokens(train_src[0], 131072))
write_shard(val_out, read_tokens(val_src[0], 65536))
print(f"created smoke dataset: {out_root}")
PY

DATA_GLOB="${DATA_PATH_SMOKE}/fineweb_train_*.bin" \
  "${REPO_ROOT}/Nitrust/scripts/medusa_nitrust_preflight.sh"

SUMMARY_TSV="${RESULT_ROOT}/summary.tsv"
printf "run_id\ttag\troundtrip_bpb\tsliding_bpb\tpeak_mem_mib\tstatus\n" > "${SUMMARY_TSV}"

COMMON_ENV=(
  "SEED=${SEED}"
  "DATA_PATH=${DATA_PATH_SMOKE}"
  "NGRAM_EVAL_ORDER=0"
  "USE_CRAWLER=1"
  "NUM_FLAT_LAYERS=4"
  "NUM_CRAWLER_LAYERS=1"
  "DELTA_NET_HEADS=0"
  "SKIP_EMA=1"
  "SKIP_GPTQ=1"
  "LOOP_AWARE_GPTQ=0"
  "MAX_WALLCLOCK_SECONDS=30"
  "ITERATIONS=20"
  "WARMUP_STEPS=0"
  "VAL_LOSS_EVERY=0"
  "TRAIN_LOG_EVERY=5"
  "TRAIN_BATCH_TOKENS=16384"
  "TRAIN_SEQ_LEN=256"
  "EVAL_SEQ_LEN=256"
  "VAL_BATCH_SIZE=32768"
  "COMPILE_ENABLED=0"
  "TORCHDYNAMO_OPTIMIZE_DDP=0"
  "COMPILE_FULLGRAPH=0"
  "NITRUST_ENABLE=1"
  "NITRUST_STRICT=1"
  "NITRUST_SO_PATH=${SO_PATH}"
)

run_case() {
  local tag="$1"
  shift
  local run_id="${tag}_$(date +%Y%m%d_%H%M%S)"
  local log_file="${RESULT_ROOT}/${tag}.log"
  local extra_env=("$@")

  echo "=== ${tag} ==="
  set +e
  env "${COMMON_ENV[@]}" "${extra_env[@]}" "RUN_ID=${run_id}" \
    python3 "${REPO_ROOT}/experiments/Medusa/train_gpt.py" 2>&1 | tee "${log_file}"
  local rc=$?
  set -e

  python3 - "${run_id}" "${tag}" "${log_file}" "${rc}" >> "${SUMMARY_TSV}" <<'PY'
import re
import sys
from pathlib import Path

run_id, tag, log_path, rc = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
text = log_path.read_text(encoding="utf-8", errors="replace")

def extract(pattern: str):
    m = re.findall(pattern, text, flags=re.MULTILINE)
    return m[-1] if m else ""

roundtrip = extract(r"final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)")
sliding = extract(r"final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)")
peak = extract(r"peak memory allocated: ([0-9]+) MiB")
status = "ok" if rc == 0 else f"fail({rc})"
print(f"{run_id}\t{tag}\t{roundtrip}\t{sliding}\t{peak}\t{status}")
PY
}

run_case "CL1-00_baseline" "CRAWLER_LOOPS=4" "INST_DIM=32" "CRAWLER_MLP_MULT=4.0" "CRAWLER_QUANT_INT8=1"
run_case "CL1-01_loops3" "CRAWLER_LOOPS=3" "INST_DIM=32" "CRAWLER_MLP_MULT=4.0" "CRAWLER_QUANT_INT8=1"
run_case "CL1-03_inst0" "CRAWLER_LOOPS=4" "INST_DIM=0" "CRAWLER_MLP_MULT=4.0" "CRAWLER_QUANT_INT8=1"
run_case "CL1-06_mlp3" "CRAWLER_LOOPS=4" "INST_DIM=32" "CRAWLER_MLP_MULT=3.0" "CRAWLER_QUANT_INT8=1"

echo "crawler leg1 smoke summary: ${SUMMARY_TSV}"
