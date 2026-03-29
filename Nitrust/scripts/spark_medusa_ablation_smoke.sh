#!/usr/bin/env bash
set -euo pipefail

# Legacy entrypoint retained for compatibility.
# Active smoke lane is crawler-only (`spark_crawler_leg1_smoke.sh`).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "[deprecation] spark_medusa_ablation_smoke.sh now forwards to crawler-only smoke."
exec "${SCRIPT_DIR}/spark_crawler_leg1_smoke.sh" "$@"
