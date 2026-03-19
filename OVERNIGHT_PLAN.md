# Parameter Golf — Overnight Autoresearch Plan
**2026-03-19 · Octavian · DGX Spark**

---

## Why Previous Attempts Failed

1. **Fractal/weight-sharing**: Better BPB per parameter but **2× slower per step** (333ms vs 167ms). In a wallclock-limited competition, step throughput matters as much as per-step quality. Half the steps = lost.
2. **AttnRes on baseline**: Correct idea (keep speed, improve architecture) but we got stuck debugging torch.compile/inductor cache aliasing and never got a clean number.
3. **Gravity**: Cool concept but adds noise at low step counts. Model learned to basically turn it off.

## The New Angle: Hyperparameter & Schedule Optimization

**Key insight**: Nobody has beaten the baseline yet. The baseline architecture is already good (inherited from modded-nanogpt). The fastest path to a leaderboard entry isn't a new architecture — it's **squeezing more juice out of the existing one** through systematic hyperparameter optimization.

The baseline has ~15 tunable knobs that were set to "reasonable defaults," not optimized for the specific 16MB/10-minute constraint:
- LR schedule (warmup steps, decay shape, cooldown fraction)
- Muon backend steps, momentum warmup
- Embedding LR vs matrix LR ratio
- Model dimensions within the 16MB budget (dim, layers, heads, KV heads, MLP mult)
- Logit softcap value
- Vocab size trade-offs
- Batch token count
- Tied embedding init std

**This is the classic "tune, don't redesign" approach** — the same one that won most modded-nanogpt records. Architecture tricks got ~45% of the speedup, but Muon optimization + schedule tuning got ~77%.

## Experiment Design

### Phase 1: Dimension Sweep (does wider/shallower beat 9×512?)
All within 16MB artifact budget. Each run: 300 steps local, ~2 min.

| Exp | layers | dim | heads | kv_heads | mlp_mult | Expected params |
|-----|--------|-----|-------|----------|----------|-----------------|
| D1  | 9      | 512 | 8     | 4        | 2        | 17.0M (baseline) |
| D2  | 7      | 576 | 8     | 4        | 2        | ~16.5M |
| D3  | 8      | 544 | 8     | 4        | 2        | ~16.8M |
| D4  | 10     | 480 | 8     | 4        | 2        | ~16.9M |
| D5  | 6      | 640 | 8     | 4        | 2        | ~16.2M |
| D6  | 9      | 512 | 8     | 4        | 3        | ~18M (test if mlp_mult=3 fits) |
| D7  | 12     | 448 | 8     | 4        | 2        | ~16.8M (deeper) |

### Phase 2: LR Schedule Sweep (on best dim from Phase 1)
| Exp | warmup | matrix_lr | embed_lr | scalar_lr | muon_steps |
|-----|--------|-----------|----------|-----------|------------|
| L1  | 20     | 0.04      | 0.05     | 0.04      | 5 (baseline) |
| L2  | 10     | 0.04      | 0.05     | 0.04      | 5 |
| L3  | 30     | 0.04      | 0.05     | 0.04      | 5 |
| L4  | 20     | 0.06      | 0.05     | 0.06      | 5 |
| L5  | 20     | 0.08      | 0.05     | 0.08      | 5 |
| L6  | 20     | 0.04      | 0.08     | 0.04      | 5 |
| L7  | 20     | 0.04      | 0.05     | 0.04      | 8 |
| L8  | 20     | 0.04      | 0.05     | 0.04      | 3 |

### Phase 3: Softcap & Init Sweep
| Exp | softcap | tied_embed_init_std |
|-----|---------|---------------------|
| S1  | 30.0    | 0.005 (baseline) |
| S2  | 15.0    | 0.005 |
| S3  | 50.0    | 0.005 |
| S4  | 30.0    | 0.01 |
| S5  | 30.0    | 0.002 |

### Phase 4: Combine Winners
Take the best from each phase, combine, run longer (600+ steps).

### Phase 5: Ship to Cloud
Best config → RunPod 1×H100 → verify BPB → if it beats 1.2244 by ≥0.005, submit PR.

## Local Execution Details

- Script: `train_local.py --mode baseline` with env var overrides
- Data: FineWeb sp1024, 1 train shard (already downloaded)
- Eval: 1M val tokens (fast proxy), full val for winners only
- Steps: 300 per experiment (~2 min each on Spark)
- Metric: val_bpb (lower is better)
- Log: append results to `OVERNIGHT_RESULTS.md`

## Runner Script

The cron job runs `scripts/overnight_sweep.sh` which:
1. Iterates through experiment configs
2. Runs each via `train_local.py`
3. Parses val_bpb from stdout
4. Appends to OVERNIGHT_RESULTS.md
5. Identifies best config per phase before starting next phase

## Success Criteria

- Find a config that beats baseline val_bpb locally by ≥ 0.01
- Relative ordering transfers from Spark to H100
- Ship to cloud for verification

---

*Plan authored by Octavian · 2026-03-19 01:15 CDT*
