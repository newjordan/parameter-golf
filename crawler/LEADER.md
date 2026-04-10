# Crawler SOTA — Current Leader

Score:  1.13541288 BPB (seed 444) | mean 1.13643599 (seeds 444/300/4)
Size:   15,902,698 bytes (max across seeds 444/300/4)
Date:   2026-04-10
Leg:    crawler/2026-04-09_Trapper_Keeper_1/
Run:    SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-09_Trapper_Keeper_1/run_7f3c_brotli_gptq_pyminify_prune.sh

## Architecture
TK1 recovered legal 7F+3C
NUM_FLAT_LAYERS=7 | NUM_CRAWLER_LAYERS=3 | CRAWLER_LOOPS=3 | INST_DIM=32
COMPILE_FULLGRAPH=1 | SKIP_GPTQ=0 | LOOP_AWARE_GPTQ=1 | RUNTIME_PYMINIFY=1
SELECTIVE_PRUNE_ENABLE=1 | CRAWLER_LOOP_ROPE_SCALES=9,1,1 | ~149ms/step on 8xH100 | ~4,000 steps in 600s

## Seeds
| Seed | BPB exact       | Size       | Status                               |
|------|-----------------|------------|--------------------------------------|
| 444  | 1.13541288      | 15,902,698 | current best seed-444 full run |
| 300  | 1.13853446      | 15,851,974 | confirmation seed passes vs prior leader |
| 4    | 1.13536063      | 15,844,157 | required third seed passes |
| mean | 1.13643599      | 15,902,698 | max bytes across all three legal seeds |

Reference metrics source:
`crawler/2026-04-09_Trapper_Keeper_1/ablation.md`

Ouroboros note:
PR #1283/#1308 reports stronger results on a separate submission lineage.
Those record folders are not present in current `TEST_LAB` tree, so TK1 recovered legal is the in-tree leader until that lineage is imported/reconciled.

## Promotion Gate
Next bar: beat 1.13541288 on seed 444 with artifact <= 16,000,000 bytes, then confirm on seeds 300 and 4.
One variable changed per leg. Gate (1-GPU, 2000 steps) before any 8x run.
