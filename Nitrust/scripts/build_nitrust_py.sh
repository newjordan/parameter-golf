#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SO_PATH="${SO_PATH:-${REPO_ROOT}/Nitrust/rust/target/release/libnitrust_py.so}"
export SO_PATH

cd "${REPO_ROOT}/Nitrust/rust"
cargo build -p nitrust-py --release

python3 - <<'PY'
import importlib.util
from pathlib import Path
import os

so_path = Path(os.environ["SO_PATH"]).resolve()
spec = importlib.util.spec_from_file_location("nitrust_py", so_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to create import spec for {so_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for fn in ("mmap_read_tokens", "build_lm_batch"):
    if not hasattr(mod, fn):
        raise RuntimeError(f"missing function in nitrust_py: {fn}")
print(f"nitrust_py import smoke OK: {so_path}")
PY

echo "built: ${SO_PATH}"
