# Suite 3: Leaderboard Killer — 8xH100
**Date:** 2026-03-20
**Branch:** experiments/sliding-window-int6
**Script:** scripts/run_leaderboard_killer.sh
**Hardware:** 8xH100 80GB (Vast.ai)
**Wallclock cap:** 600s per run

## Results

| Run | Config | Params | Steps | Pre-quant | Quant BPB | Artifact |
|-----|--------|--------|-------|-----------|-----------|----------|
| 1 | MLP1344 + FP16embed + SmearGate + BigramHash + OrthoInit + earlyQAT25 + stride64 + MuonWD | 20.6M | 11134 | 1.2022 | **1.1725** | **15.58MB** |

## Key Findings

1. **First valid submission**: 15.58MB fits under 16MB limit with 424KB headroom
2. **SmearGate + BigramHash + OrthoInit all active**: full leaderboard stack
3. **Quant BPB 1.1725**: would place ~7th-8th on validated leaderboard (top is 1.1483)
4. **MLP shrink cost**: 1.1725 vs 1.1693 (oversized run) — lost 0.003 BPB from MLP 1344 vs 1536
5. **424KB headroom**: room to bump MLP_HIDDEN to ~1472 for next run
6. **53.87ms/step**: slightly slower than MLP3x (49ms) due to BigramHash overhead
7. **Fewer steps**: 11134 vs 12173 (MLP3x) due to slower per-step time

## Comparison Across All Sessions

| Config | Quant BPB | Artifact | Valid? |
|--------|-----------|----------|--------|
| 9L MLP2x baseline | 1.3306 | 12.5MB | yes |
| 9L MLP2x slide512 | 1.3112 | 12.5MB | yes |
| 9L MLP3x slide512 | 1.2681 | 15.9MB | yes |
| 9L MLP3x earlyQAT slide512 | 1.2607 | 15.9MB | yes |
| 11L MLP2x slide512 | 1.2784 | 15.2MB | yes |
| 9L MLP3x FP16embed stride64 | 1.1693 | 16.34MB | NO |
| **9L MLP1344 full stack stride64** | **1.1725** | **15.58MB** | **yes** |

## Next Steps

- Bump MLP_HIDDEN from 1344 to ~1472 to use 424KB headroom
- Close gap from 1.1725 toward 1.1693
