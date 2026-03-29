# crawler_looped — Hypothesis

## Problem

Crawler shared blocks collapse to a fixed point across K loops.
Each loop produces ~0% additional BPB improvement (recursion = 0% per-step).
The K passes through the shared block converge to the same attractor,
making the loops redundant.

Previous attempt (DeltaNet): collapsed into memorization — learned to
map token identity → correct answer rather than learning generalizable
correction of compression error.

## Proposed Fix: Per-Loop LoRA Adapters

Each crawler loop k gets a unique low-rank correction on the MLP up-projection:

```
h = W_fc @ x              ← shared base (all loops)
  + A_k @ (B_k @ x)      ← loop-specific rank-r correction
```

**Why this breaks the collapse:**
- Each loop k's adapter (A_k, B_k) only receives gradient from loop k's forward pass
- The base weights W_fc still optimize for the average — but each loop has a unique
  "flavor" that the optimizer can differentiate
- A_k is zero-initialized → adapter starts as identity, training is smooth
- The fixed-point symmetry is broken: loop 1 and loop 2 have different optimization
  targets, so they can't converge to the same attractor

## The Dial: LOOP_ADAPTER_RANK

| Rank | Params (K=4 loops) | Artifact cost (int6) | Notes |
|------|-------------------|----------------------|-------|
| 0 | 0 | 0 | Baseline — collapses |
| 1 | ~10K | ~4KB | Minimal test |
| 4 | ~40K | ~15KB | First real test |
| 8 | ~82K | ~30KB | Higher capacity |

All negligible vs. total artifact size.

## Test Protocol

1. Baseline: `LOOP_ADAPTER_RANK=0` (identical to Crawler_Leg_1)
2. Test: `LOOP_ADAPTER_RANK=4`

Compare:
- Final BPB at 600s wall-clock (same seed=1337)
- Per-loop BPB contribution (is loop 2 > loop 1? loop 3 > loop 2?)

Signal: if per-loop BPB is monotonically decreasing, collapse is broken.
If final BPB improves over baseline, the fix is real.

## Ablation Grid

```
LOOP_ADAPTER_RANK=1   → does any rank break collapse?
LOOP_ADAPTER_RANK=4   → primary test
LOOP_ADAPTER_RANK=8   → diminishing returns check
```
