# Crawler-Only Leg 1 Tracker
Date: 2026-03-29

## Mainline / Sandbox Split
- Mainline crawler run (Delta OFF): `experiments/Crawler_Leg_1/run.sh`
- Compatibility alias: `experiments/Medusa/run.sh`
- Delta quarantine sandbox: `experiments/Medusa/run_delta_sandbox.sh`

## Research Rule
- For Leg 1, all runs must keep `DELTA_NET_HEADS=0`.

## Immediate Runner

```bash
bash Nitrust/scripts/spark_crawler_leg1_smoke.sh
```

8xH100 launcher (with Nitrust preflight):

```bash
bash Nitrust/scripts/run_crawler_nitrust_8xh100.sh
```

Single-GPU package:

```bash
MODE=full bash Nitrust/scripts/run_crawler_nitrust_1gpu.sh
```

Outputs:
- `results/crawler_leg1_smoke_<timestamp>/summary.tsv`

## Priority Ablations
1. `CRAWLER_LOOPS`: 3 / 4 / 5
2. `INST_DIM`: 0 / 16 / 32 / 64
3. `CRAWLER_MLP_MULT`: 3.0 / 4.0 / 5.0
4. `CRAWLER_QUANT_INT8`: 0 / 1
5. Flat/Crawler split: `(NUM_FLAT_LAYERS, NUM_CRAWLER_LAYERS)` variations
