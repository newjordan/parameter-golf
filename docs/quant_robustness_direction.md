# Quant Robustness Direction
Date: 2026-03-19
Branch intent: focus experiments on reducing post-training quantization loss while preserving 10-minute track competitiveness.

## Why This Direction
- The 4h reference run shows strong pre-quant BPB (`1.1749`) but larger post-quant penalty (`+0.0325 BPB`) than the 10-min baseline (`+0.0072 BPB`).
- Closing this quantization gap can convert existing pre-quant gains into scored gains.

## Core Hypothesis
- The current post-training int8+zlib flow leaves recoverable quality on the table.
- Training for quant robustness (and/or selectively preserving sensitive tensors) should reduce the pre/post BPB gap.

## Initial Experiment Lanes
1. Keep-float sensitivity sweep:
   - Expand or tune `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS`
   - Tune `INT8_KEEP_FLOAT_MAX_NUMEL`
   - Goal: retain fragile tensors in float with minimal size overhead
2. Quant clipping sweep:
   - Tune `INT8_CLIP_PERCENTILE`
   - Goal: reduce quantization distortion in heavy-tailed tensors
3. Lightweight quant-aware finetune phase:
   - Short final phase with simulated quantization perturbation before export
   - Goal: improve robustness without major wall-clock regression

## Success Criteria
- Primary: lower `final_int8_zlib_roundtrip_exact val_bpb`
- Secondary:
  - artifact remains `< 16,000,000 bytes`
  - no runtime regressions that break 10-minute reproducibility

## Risks
- Keeping too many tensors float can exceed size budget.
- Aggressive clipping changes may help some layers but hurt others.
- Added quant-aware phase can consume wall-clock budget.
