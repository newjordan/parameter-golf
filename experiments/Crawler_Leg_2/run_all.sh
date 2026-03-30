#!/bin/bash
set -euo pipefail
# CRAWLER_LEG_2 — Additivity ablation: loops=3 + mlp=5.0 combined
# Then stack LOOP_AWARE_GPTQ + COMPILE_FULLGRAPH on top.
#
# 7 arms, 600s each, 1×H100 (~70 min total)
#
# Usage:
#   NPROC_PER_NODE=1 bash experiments/Crawler_Leg_2/run_all.sh
#   NPROC_PER_NODE=8 bash experiments/Crawler_Leg_2/run_all.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/flash-attention/hopper:${PYTHONPATH:-}"

SEED="${SEED:-1337}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NITRUST_ENABLE="${NITRUST_ENABLE:-0}"
NITRUST_STRICT="${NITRUST_STRICT:-0}"
NITRUST_SO_PATH="${NITRUST_SO_PATH:-Nitrust/rust/target/release/libnitrust_py.so}"

RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_DIR}"

RUN_DATE="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${RESULTS_DIR}/summary_${RUN_DATE}.txt"

echo "============================================"
echo "  CRAWLER_LEG_2 — Additivity Ablation"
echo "  loops=3 + mlp=5.0 combined, then stack GPTQ+compile"
echo "  Seed: ${SEED}  GPUs: ${NPROC_PER_NODE}  Wallclock: 600s/arm"
echo "  Arms: CL2-00 through CL2-06 (7 total)"
echo "  NITRUST_ENABLE=${NITRUST_ENABLE}"
echo "============================================"
echo ""

# -------------------------------------------------------------------
# Preflight checks (same as Bandit)
# -------------------------------------------------------------------
echo "[preflight] checking zstandard..."
python3 -c "import zstandard; print(f'  zstandard {zstandard.__version__} OK')" 2>/dev/null \
    || echo "  WARNING: zstandard not found"

echo "[preflight] patching torch inductor AttrsDescriptor bug (if present)..."
python3 -c "
import importlib.util, pathlib
spec = importlib.util.find_spec('torch._inductor.runtime.hints')
if spec and spec.origin:
    p = pathlib.Path(spec.origin)
    txt = p.read_text()
    old = 'attr_desc_fields = {f.name for f in fields(AttrsDescriptor)}'
    if old in txt:
        import attr
        new = 'import attr as _attr; attr_desc_fields = {f.name for f in _attr.fields(AttrsDescriptor)}'
        p.write_text(txt.replace(old, new))
        print('  patched OK')
    else:
        print('  no patch needed')
" 2>/dev/null || echo "  WARNING: could not patch hints.py"

echo "[preflight] checking flash_attn..."
python3 -c "
try:
    import flash_attn_interface; print('  FA3 (hopper) OK')
except ImportError:
    import flash_attn; v=flash_attn.__version__
    if v.startswith('3'): print(f'  FA3 v{v} OK')
    else: print(f'  WARNING: FA{v[0]} detected — want FA3')
" 2>/dev/null || echo "  WARNING: no flash_attn found"

# -------------------------------------------------------------------
# run_arm <arm_id> <label> <extra env overrides...>
# -------------------------------------------------------------------
run_arm() {
    local arm_id="$1"
    local label="$2"
    shift 2
    # remaining args are KEY=VALUE overrides

    local log="${RESULTS_DIR}/${arm_id}_${RUN_DATE}.log"

    echo "================================================================"
    echo "  ARM ${arm_id}  —  ${label}"
    echo "  Log: ${log}"
    echo "================================================================"

    env \
        SEED="${SEED}" \
        MAX_WALLCLOCK_SECONDS=600 \
        WARMDOWN_ITERS=2000 \
        COMPLEMENT_ALPHA=0 \
        XSA_LAST_N=11 \
        BIGRAM_VOCAB_SIZE=2048 \
        ROPE_DIMS=16 \
        SWA_EVERY=50 \
        MTP_NUM_HEADS=0 \
        LATE_QAT_THRESHOLD=0 \
        MATRIX_LR=0.03 \
        TORCHDYNAMO_OPTIMIZE_DDP=0 \
        COMPILE_FULLGRAPH=0 \
        NGRAM_EVAL_ORDER=0 \
        USE_CRAWLER=1 \
        NUM_FLAT_LAYERS=4 \
        NUM_CRAWLER_LAYERS=1 \
        CRAWLER_LOOPS=4 \
        CRAWLER_MLP_MULT=4.0 \
        INST_DIM=32 \
        CRAWLER_QUANT_INT8=1 \
        DELTA_NET_HEADS=0 \
        SKIP_EMA=1 \
        SKIP_GPTQ=1 \
        LOOP_AWARE_GPTQ=0 \
        NITRUST_ENABLE="${NITRUST_ENABLE}" \
        NITRUST_STRICT="${NITRUST_STRICT}" \
        NITRUST_SO_PATH="${NITRUST_SO_PATH}" \
        "$@" \
        torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
            "${SCRIPT_DIR}/train_gpt.py" \
        2>&1 | tee "${log}"

    # extract final val_bpb (last val eval line before stop)
    local val_bpb
    val_bpb=$(grep -oP 'val_bpb:\K[0-9.]+' "${log}" | tail -1)
    local steps
    steps=$(grep -oP 'stopping_early.*step:\K[0-9]+' "${log}" | tail -1 \
            || grep -oP 'step:(\K[0-9]+)/20000 val_loss' "${log}" | tail -1 \
            || echo "?")

    echo "${arm_id}|${label}|${steps}|${val_bpb}" >> "${SUMMARY}.tmp"
    echo "  -> val_bpb: ${val_bpb}  steps: ${steps}"
    echo ""
}

# -------------------------------------------------------------------
# Arms
# -------------------------------------------------------------------

# CL2-00: Leg 1 baseline (loops=4, mlp=4.0)
run_arm CL2-00 "baseline (loops=4 mlp=4.0)"

# CL2-01: loops=3 only (reproduce Leg 1 single win)
run_arm CL2-01 "loops=3 only" \
    CRAWLER_LOOPS=3

# CL2-02: mlp=5.0 only (reproduce Leg 1 single win)
run_arm CL2-02 "mlp=5.0 only" \
    CRAWLER_MLP_MULT=5.0

# CL2-03: KEY ARM — loops=3 + mlp=5.0 combined (additivity test)
run_arm CL2-03 "loops=3 + mlp=5.0 (additivity)" \
    CRAWLER_LOOPS=3 \
    CRAWLER_MLP_MULT=5.0

# CL2-04: loops=3 + mlp=5.0 + LOOP_AWARE_GPTQ=1
# NOTE: SKIP_GPTQ still 1 here — LOOP_AWARE_GPTQ affects training behavior
#       even when GPTQ is skipped (2-phase Hessian calibration awareness)
run_arm CL2-04 "loops=3+mlp=5.0+GPTQ_AWARE" \
    CRAWLER_LOOPS=3 \
    CRAWLER_MLP_MULT=5.0 \
    LOOP_AWARE_GPTQ=1

# CL2-05: loops=3 + mlp=5.0 + COMPILE_FULLGRAPH=1
run_arm CL2-05 "loops=3+mlp=5.0+COMPILE" \
    CRAWLER_LOOPS=3 \
    CRAWLER_MLP_MULT=5.0 \
    COMPILE_FULLGRAPH=1

# CL2-06: FULL STACK — loops=3 + mlp=5.0 + GPTQ + compile
run_arm CL2-06 "FULL STACK (loops3+mlp5+GPTQ+compile)" \
    CRAWLER_LOOPS=3 \
    CRAWLER_MLP_MULT=5.0 \
    LOOP_AWARE_GPTQ=1 \
    COMPILE_FULLGRAPH=1

# -------------------------------------------------------------------
# Summary table
# -------------------------------------------------------------------
BASELINE_BPB=$(grep "^CL2-00|" "${SUMMARY}.tmp" | cut -d'|' -f4)

echo "================================================================"
echo "  CRAWLER_LEG_2 COMPLETE — ${RUN_DATE}"
echo "  Seed: ${SEED}  Baseline val_bpb: ${BASELINE_BPB}"
echo "================================================================"
printf "%-10s %-40s %6s %8s %9s\n" "ARM" "LABEL" "STEPS" "VAL_BPB" "DELTA"
printf "%-10s %-40s %6s %8s %9s\n" "---" "-----" "-----" "-------" "-----"

while IFS='|' read -r arm label steps bpb; do
    if [[ -n "${BASELINE_BPB}" && -n "${bpb}" && "${arm}" != "CL2-00" ]]; then
        delta=$(python3 -c "print(f'{float(\"${bpb}\")-float(\"${BASELINE_BPB}\"):+.4f}')" 2>/dev/null || echo "?")
    else
        delta="—"
    fi
    printf "%-10s %-40s %6s %8s %9s\n" "${arm}" "${label}" "${steps}" "${bpb}" "${delta}"
done < "${SUMMARY}.tmp"

echo ""
echo "  Additivity check:"
echo "    CL2-01 delta (loops=3) + CL2-02 delta (mlp=5.0) vs CL2-03 delta (combined)"
echo "    If CL2-03 ≈ CL2-01 + CL2-02 → wins are additive"
echo ""
echo "================================================================"
echo "  Full logs: ${RESULTS_DIR}/"
echo "================================================================"

mv "${SUMMARY}.tmp" "${SUMMARY}"
