# Suite 6: Stinky Frost V2 — 8xH100
**Date:** 2026-03-21
**Branch:** experiments/stinky-frost-v2
**Script:** scripts/run_stinky_frost_v2_8xh100.sh
**Hardware:** 8xH100 80GB (RunPod)
**Wallclock cap:** 600s

## Results

| Run | Config | Params | Steps | Pre-quant BPB | Artifact | Valid? |
|-----|--------|--------|-------|---------------|----------|--------|
| V2 | 11L/512d MLP3x + XSA(3) + zstd-22 + SWA + WD0.04 + all techniques | 27.9M | 8130 | 1.1877 | 20.7MB | **NO — 4.7MB over** |

## Key Findings

1. **Pre-quant BPB 1.1877 is strong** — beats v1's 1.2022 significantly
2. **BUSTED 16MB by 4.7MB** — 11L + MLP3x (1536) + FP16 embed + BigramHash(10240) = too many params
3. **74ms/step** — 8130 steps in 600s
4. **SWA only got 3 checkpoints** (started at step 8000, ended at 8130)
5. **Convergence was still accelerating**: 1.2744 → 1.1877 in last 1130 steps (warmdown effect)
6. **Code size 102KB** — bloated, needs trimming

## Val BPB Trajectory

| Step | Val BPB | Notes |
|------|---------|-------|
| 500 | 1.4556 | |
| 1000 | 1.3684 | |
| 2000 | 1.3210 | |
| 3000 | 1.3022 | |
| 4000 | 1.2919 | |
| 5000 | 1.2861 | QAT activated |
| 6000 | 1.2832 | |
| 7000 | 1.2744 | |
| 7500 | 1.2377 | Warmdown kicking in |
| 8000 | 1.1961 | SWA started |
| 8130 | 1.1877 | Wallclock stop |

## Size Problem Analysis

Need to cut ~5MB to fit 16MB. Options:
- Reduce MLP from 1536 to ~1024-1100 for 11L
- Go to 10L with MLP ~1200
- Use int5 for MLP (but hurts at 11L per competition data)
- Reduce BigramHash from 10240 to 4096 (saves ~0.5MB)
- Trim code from 102KB to ~50KB (saves ~50KB)
- Remove FP16 embed (saves ~0.65MB but costs 0.1 BPB)
