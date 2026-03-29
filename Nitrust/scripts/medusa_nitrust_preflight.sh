#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SO_PATH="${SO_PATH:-${REPO_ROOT}/Nitrust/rust/target/release/libnitrust_py.so}"
DATA_GLOB="${DATA_GLOB:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin}"
export SO_PATH DATA_GLOB

"${SCRIPT_DIR}/build_nitrust_py.sh"

python3 - <<'PY'
import glob
import importlib.util
import os
from pathlib import Path
import numpy as np

so_path = Path(os.environ["SO_PATH"]).resolve()
data_glob = os.environ["DATA_GLOB"]
files = sorted(glob.glob(data_glob))
if not files:
    raise FileNotFoundError(f"no train shards found for DATA_GLOB={data_glob}")
file_path = Path(files[0])

spec = importlib.util.spec_from_file_location("nitrust_py", so_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to create import spec for {so_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

header = np.fromfile(file_path, dtype="<i4", count=256)
if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
    raise ValueError(f"unexpected shard header for {file_path}")
num_tokens = int(header[2])

for off, ln in ((0, 16), (12345, 32), (num_tokens - 64, 64)):
    py = np.fromfile(file_path, dtype="<u2", count=ln, offset=256 * 4 + off * 2)
    rs = np.asarray(mod.mmap_read_tokens(str(file_path), off, ln), dtype=np.uint16)
    if not np.array_equal(py, rs):
        raise RuntimeError(f"parity failure at offset={off} len={ln}")

print(f"preflight OK shard={file_path} total_tokens={num_tokens}")
PY

echo "medusa nitrust preflight: PASS"
