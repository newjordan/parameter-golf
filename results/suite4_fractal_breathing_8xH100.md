# Suite 4: Fractal Breathing — 8xH100
**Date:** 2026-03-20
**Branch:** experiments/sliding-window-int6
**Script:** scripts/run_fractal_breathing.sh
**Hardware:** 8xH100 80GB (Vast.ai)
**Wallclock cap:** 600s

## Results

| Run | Config | Params | Steps | Pre-quant | Quant BPB | Artifact |
|-----|--------|--------|-------|-----------|-----------|----------|
| 1 | Fractal 3×3, 960d, full/cheap/cheap + full stack | 20.1M | 7137 | 1.2655 | 1.2373 | 15.37MB |

## Verdict

**Fractal is too slow for the 10-min wallclock.** 84ms/step = only 7137 steps vs 11134 for non-fractal. The 0.065 BPB gap to The Stinky Frost Recipe (1.1725) is entirely from lost training steps.

The breathing pattern (full,cheap,cheap) helped — without it, fractal would be ~100ms/step and even fewer steps. But it's not enough.

## Conclusion

Fractal depth recurrence is a sound architectural idea but needs more training time than the competition allows. The 10-minute wallclock constraint favors fast, simple architectures (standard transformer + technique stacking) over novel architectures with higher per-step cost.

For a competition with longer training time or smaller models, fractal could be competitive.
