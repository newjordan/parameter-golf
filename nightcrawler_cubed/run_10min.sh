#!/bin/bash
set -euo pipefail
# Nightcrawler Cubed (7F+3C) — current legal 10-minute crawler path
#
# Thin alias for the canonical 7F+3C runner.
# The exact model/export stack lives in run.sh so there is only one source of truth.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/run.sh"
