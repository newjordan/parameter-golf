#!/usr/bin/env bash
set -euo pipefail

# Canonical crawler-only launcher.
# Delegates to the legacy-named compatibility script.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_medusa_nitrust_8xh100.sh" "$@"
