#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "  JR-03 — Full MLP Fusion Candidate"
echo "  target kernel_mode=triton_fused_mlp"
echo "  status=setup only; implementation not landed yet"
echo "============================================"

echo "ERROR: triton_fused_mlp is not implemented in this branch yet."
echo "Next step is kernel + bench integration under experiments/Junkyard_Rat_MLP/triton/."
exit 2
