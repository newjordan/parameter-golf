# Suite 3: Leaderboard Killer — 8xH100
**Date:** 2026-03-20
**Branch:** experiments/sliding-window-int6
**Script:** scripts/run_leaderboard_killer.sh
**Hardware:** 8xH100 80GB (Vast.ai)
**Wallclock cap:** 600s per run

## Results

| Run | Config | Params | Steps | Pre-quant | Quant BPB | Artifact | Valid? |
|-----|--------|--------|-------|-----------|-----------|----------|--------|
| v1 | MLP1344 + FP16embed + SmearGate + BigramHash + OrthoInit + earlyQAT25 + stride64 + MuonWD0.01 | 20.6M | 11134 | 1.2022 | **1.1725** | 15.58MB | YES |
| v2 | MLP1472 + same stack | 21.8M | 11114 | 1.1982 | **1.1686** | 16.42MB | NO |

## Key Findings

1. **v1 (MLP1344) is valid**: 15.58MB fits under 16MB with 424KB headroom
2. **v2 (MLP1472) busts limit**: 16.42MB, 422KB over — but best quant BPB at 1.1686
3. **MLP1408 is the sweet spot**: splitting 1344/1472 should land ~16.0MB (v4 pending)
4. **SmearGate + BigramHash + OrthoInit all active**: full leaderboard stack
5. **v2 quant BPB 1.1686 beats our previous best** (1.1693) by 0.0007
6. **~54ms/step**: slightly slower than MLP3x alone (49ms) due to BigramHash/SmearGate
7. **Fewer steps**: ~11100 vs 12173 (MLP3x) due to slower per-step time

## Comparison Across All Sessions

| Config | Quant BPB | Artifact | Valid? |
|--------|-----------|----------|--------|
| 9L MLP2x baseline | 1.3306 | 12.5MB | yes |
| 9L MLP2x slide512 | 1.3112 | 12.5MB | yes |
| 9L MLP3x slide512 | 1.2681 | 15.9MB | yes |
| 9L MLP3x earlyQAT slide512 | 1.2607 | 15.9MB | yes |
| 11L MLP2x slide512 | 1.2784 | 15.2MB | yes |
| 9L MLP3x FP16embed stride64 | 1.1693 | 16.34MB | NO |
| 9L MLP1344 full stack stride64 | 1.1725 | 15.58MB | yes |
| **9L MLP1472 full stack stride64** | **1.1686** | **16.42MB** | **NO** |

## Next Steps

- v4: MLP_HIDDEN=1408 + SWA every 50 steps + MuonWD=0.04 (matching #1 entry)
- SWA should improve quant gap by smoothing weight distributions
- Target: get 1.1686-level BPB while fitting under 16MB
