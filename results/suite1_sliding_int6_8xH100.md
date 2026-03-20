# Suite 1: Sliding Window + Int6 QAT — 8xH100
**Date:** 2026-03-20
**Branch:** experiments/sliding-window-int6
**Script:** scripts/run_sliding_int6_1xh100.sh
**Hardware:** 8xH100 80GB (Vast.ai)
**Wallclock cap:** 600s per run

## Results

| Run | Config | Params | Steps | TTT BPB | Quant BPB | Artifact Size |
|-----|--------|--------|-------|---------|-----------|---------------|
| 1 | 9L/512d MLP2x, no slide | 17.1M | 13159 | 1.2686 | 1.3306 | 12.5MB |
| 2 | 9L/512d MLP2x, slide512 | 17.1M | 13159 | 1.2780 | 1.3112 | 12.5MB |
| **3** | **9L/512d MLP3x, slide512** | **21.8M** | **12013** | **1.2409** | **1.2681** | **15.9MB** |

## Key Findings

1. **Sliding window eval is free BPB on quant**: 1.3306 -> 1.3112 (-0.019 BPB) at zero param cost
2. **MLP 3x is the biggest win**: -0.043 BPB quant, -0.028 BPB TTT vs baseline slide512
3. **MLP 3x fits in 16MB**: 15.9MB artifact, just under the limit
4. **9L/512d MLP2x wastes budget**: only 12.5MB of 16MB used — 3.5MB headroom
5. **TTT BPB slightly worse with sliding window** (1.2780 vs 1.2686) — possible interaction with TTT LoRA eval
6. **Step throughput**: ~45.6ms/step (MLP2x), ~49.9ms/step (MLP3x) — MLP3x is ~10% slower per step
7. **QAT activated at step 10000** in all runs, warmdown acceleration visible after step 12500

## Training Details

- All runs hit wallclock cap before reaching 20000 iterations
- MLP3x reached fewer steps (12013 vs 13159) due to higher per-step cost
- Peak memory: 10.1GB (MLP2x), 11.0GB (MLP3x)
- Compression ratio: ~5.2x (int6+zlib vs raw torch)

## Next Steps

- Test 11L/512d MLP2x (deeper, fill budget with layers instead of width)
- Test SmearGate on 11L config
- Test fractal 3x3 at 960d (weight sharing allows wider model)
