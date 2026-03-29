# Crawler Leg 1

Date: 2026-03-29  
Status: Active research lane (crawler-only)

## Mission
Stabilize and improve crawler behavior with DeltaNet fully quarantined from mainline runs.

Bandit is the current external SOTA reference. Leg 1 is focused on recovering crawler-only signal before any DeltaNet re-entry.

## Hard Contract
- `DELTA_NET_HEADS=0` for all Leg 1 runs.
- NGRAM evaluation stays off (`NGRAM_EVAL_ORDER=0`).
- Report model-only metrics first:
  - `final_int6_roundtrip_exact`
  - `final_int6_sliding_window_exact`

## Canonical Commands

Single run:

```bash
bash experiments/Crawler_Leg_1/run.sh
```

8xH100 with Nitrust preflight:

```bash
bash Nitrust/scripts/run_crawler_nitrust_8xh100.sh
```

Single GPU package (smoke/full/both):

```bash
MODE=full bash Nitrust/scripts/run_crawler_nitrust_1gpu.sh
```

Spark smoke ablations:

```bash
bash Nitrust/scripts/spark_crawler_leg1_smoke.sh
```

## Ablation Queue (Complexity Order)
1. Loop count: `CRAWLER_LOOPS` (3/4/5)
2. Instruction bottleneck: `INST_DIM` (0/16/32/64)
3. Shared block width: `CRAWLER_MLP_MULT` (3.0/4.0/5.0)
4. Shared block quant policy: `CRAWLER_QUANT_INT8` (0/1)
5. Flat/crawler depth split: `NUM_FLAT_LAYERS`, `NUM_CRAWLER_LAYERS`

## Output Locations
- Smoke summaries: `results/crawler_leg1_smoke_<timestamp>/summary.tsv`
- Training logs: `logs/crawler_leg1_*.log`
