# Suite 2: Best Shot — 8xH100
**Date:** 2026-03-20
**Branch:** experiments/sliding-window-int6
**Script:** scripts/run_best_shot.sh
**Hardware:** 8xH100 80GB (Vast.ai)
**Wallclock cap:** 600s per run

## Results

| Run | Config | Params | Steps | Pre-quant | Quant BPB | TTT BPB | Artifact |
|-----|--------|--------|-------|-----------|-----------|---------|----------|
| 1 | MLP3x + earlyQAT25 + stride64 + FP16embed + MuonWD | 21.8M | 12173 | 1.1969 | **1.1693** | 1.1701 | 16.34MB OVER |
| 2 | MLP3x + earlyQAT25 + stride64 (no FP16/MuonWD) | 21.8M | 12224 | 1.1977 | 1.2686 | — | 15.9MB |

## Key Findings

1. **FP16 embed is worth ~0.1 BPB on quant**: 1.2686 → 1.1693. The single biggest improvement found tonight.
2. **Without FP16 embed, stride=64 does almost nothing**: 1.2686 vs 1.2681 (stride=512 from suite 1). The bottleneck is embedding precision, not eval context.
3. **FP16 embed + stride=64 synergize**: FP16 preserves embedding quality → stride=64 can exploit richer context.
4. **Run 1 busts 16MB by 340KB**: 16,340,633 bytes vs 16,000,000 limit. Need to shrink MLP to fit.
5. **TTT LoRA slightly hurt with FP16 embed + stride64**: 1.1701 vs 1.1693 quant. Possible interaction.
6. **MuonWD effect unclear**: can't isolate from run 1 vs 2 (both changed FP16 and MuonWD).
7. **Eval with stride=64 is slow**: ~370s eval time vs ~63s for stride=512.

## Training Details

- Both runs hit wallclock cap at ~12200 steps
- QAT activated at step 5000 (25% of 20000) — confirmed working with inline env vars
- Previous `export` approach had env propagation bug — inline vars on torchrun command line is the fix
- Peak memory: 11.3GB

## Next Steps

- Shrink MLP from 1536 (3×) to 1344 to fit FP16 embed under 16MB
- Stack SmearGate + BigramHash + OrthoInit on the size-optimized config
- Built as `scripts/run_leaderboard_killer.sh`
