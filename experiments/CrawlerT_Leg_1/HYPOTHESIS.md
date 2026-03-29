# CrawlerT_Leg_1 — Hypothesis

## What We're Testing

**H**: Compiling the non-attention ops in the crawler with `mode='max-autotune'` gives
≥15% faster steps, which translates to more training steps in the 600s wall-clock budget,
which improves BPB.

## Background

The crawler's hot path per loop iteration (non-attention):

```
F.rms_norm(x)               ← separate kernel launch
* ln_scale_factor           ← elementwise
CastedLinear.fc(x)          ← weight.to(bf16) materialized + GEMM [D→4D]
F.relu(x).square()          ← 2 separate elementwise kernels
CastedLinear.proj(x)        ← weight.to(bf16) materialized + GEMM [4D→D]
* mlp_scale + residual add  ← 2 more elementwise
```

That's 6+ kernel launches per MLP pass, each touching the full `[B, T, D]` activation.
With K=4 crawler loops, that's 24+ non-attention kernel launches per forward.

Current training: `torch.compile(mode=None, fullgraph=False)` — default mode.
The Python `for loop` in `_run_crawler` forces `fullgraph=False`, limiting inductor's ability
to fuse across the loop boundary. But within each block call, inductor should still be able
to fuse elementwise ops IF given enough budget via `max-autotune`.

## Variables

- **Baseline**: `COMPILE_FULLGRAPH=0`, default compile mode (Crawler_Leg_1 config)
- **Test**: `TRITON_FUSE=1` → `torch.compile(model, mode='max-autotune', fullgraph=False)`

Everything else identical: same arch, same seed, same 600s budget.

## Phase 1 Gate (bench.py)

Before running the full 600s experiment, `bench.py` checks if `max-autotune` actually
speeds up the NormMLP kernel in isolation on this hardware. If the speedup is <10%,
the full run won't produce a meaningful BPB delta and we should investigate custom Triton
kernels instead.

**Gate**: ≥10% speedup on bench.py → proceed to Phase 2 (run.sh).

## Expected Outcome

If the hypothesis is correct:
- bench.py shows 15-25% speedup on NormMLP
- 600s run shows measurable BPB improvement vs. Crawler_Leg_1 baseline
- The improvement scales with the fraction of step time in non-attention ops

## Connection to "Removed Cylinder"

This doesn't fix the cylinder (that's architectural). But more steps in budget means
the model better optimizes the capacity it has. At the competition margin (~0.01 BPB),
a 20% step increase could matter.
