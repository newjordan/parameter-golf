# Suite 5: Fractal Progressive — 8xH100
**Date:** 2026-03-20
**Branch:** experiments/sliding-window-int6
**Script:** scripts/run_fractal_progressive.sh
**Hardware:** 8xH100 80GB (RunPod)
**Wallclock cap:** 600s

## Results

| Run | Config | Params | Steps | Pre-quant | Quant BPB | TTT BPB | Artifact |
|-----|--------|--------|-------|-----------|-----------|---------|----------|
| 1 | Fractal 3×3 960d, progressive (1→2→3 loops), full/cheap/cheap, full stack | 20.1M | 9086 | 1.2706 | 1.2438 | 1.2454 | 15.29MB |

## Comparison to Other Fractal Runs

| Config | Steps | Pre-quant | Quant BPB |
|--------|-------|-----------|-----------|
| Fractal breathing (no progressive) | 7137 | 1.2655 | 1.2373 |
| **Fractal progressive** | **9086** | **1.2706** | **1.2438** |

Progressive got 27% more steps (9086 vs 7137) but quant BPB was slightly worse (1.2438 vs 1.2373). Loop transitions caused val BPB instability — bounced from 1.45 to 2.00 during transitions at steps 3000 and 5000.

## Key Observations

1. **Progressive unrolling delivered more steps** but the hard loop cutover disrupted training
2. **Val BPB was still dropping fast at termination**: 1.3009 → 1.2706 in last 586 steps
3. **Per-step time crept up**: 42ms (1 loop) → 48ms (2 loops) → 66ms (3 loops)
4. **Not competitive with non-fractal**: 1.2438 vs Stinky Frost's 1.1725
5. **Needs smoother transitions**: gradual loop blending instead of hard cutover

## Verdict

Fractal progressive is an improvement over naive fractal (more steps) but the transition instability and inherent per-step cost make it uncompetitive in the 10-minute regime. The idea has merit for longer training or smoother transition mechanisms.
