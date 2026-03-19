# Gen_1_aTTn: AttnRes Depth Attention — Results

**Branch**: `experiments/Gen_1_aTTn`
**Date**: 2026-03-19
**Hardware**: 1×H100 (RunPod), single-process via `python train_gpt.py`

---

## Idea

Replace the U-Net skip connections (encoder-decoder with learned `skip_weights`) with **Attention Residuals** (AttnRes) from the Kimi-K2 / Moonshot paper (arxiv:2603.15031). Each layer attends over all previous layer outputs using a learned query vector per layer, computing a softmax-weighted combination as the residual stream input.

## Implementation

- `attnres_queries`: `nn.Parameter([num_layers, model_dim])` = [9, 512] = 4,608 params
- Removed `skip_weights` [4, 512] = 2,048 params (net +2,560 params)
- Per layer: `torch.stack(entries)` → RMS norm → einsum dot-product → softmax → weighted sum
- `fullgraph=True` compilation succeeded (loop unrolled, static shapes per iteration)

## Results

| Metric | AttnRes | Baseline |
|---|---|---|
| val_bpb @ step 200 | **1.6463** | ~1.35* |
| val_bpb @ step 400 | **1.5224** | **1.2244** |
| step_avg | ~800 ms | ~493 ms |
| steps in 600s budget | ~750 | ~1,200 |
| params | 17,059,912 | 17,062,472 |

*Baseline step-200 estimated from interpolation; exact number from prior runs.

## Analysis

### Why it lost (~24% worse BPB)

1. **Step throughput penalty**: ~60% slower per step (800ms vs 493ms). In a wallclock-limited competition, this means ~40% fewer training steps. The `torch.stack` + two einsum ops per layer (×9 layers × 8 grad_accum microbatches) add substantial overhead.

2. **Depth attention may need more steps to converge**: AttnRes replaces hard-coded U-Net topology with a learned routing. The queries start near-random (init `randn * 0.01`), so early training wastes steps discovering the skip structure that U-Net provides for free. With only ~750 steps in budget, this warmup cost is significant.

3. **Full-depth attention is overkill at 9 layers**: The Kimi-K2 paper targets 60+ layer models where depth attention over many layers provides real routing flexibility. At 9 layers with only 4 skip connections in the baseline, the added expressiveness doesn't justify the cost.

### Crash at step ~400

The run terminated silently after step 400 (~320s). Most likely cause: CUDA OOM from growing `torch.stack` allocations retained for backward. The total memory for cloned layer caches scales as O(L²) — sum of 1+2+...+9 = 45 tensor slots of [B, T, D] each, all kept alive for gradient computation.

### What we learned

- `torch.compile(fullgraph=True)` works with list + `torch.stack` pattern (compilation succeeded)
- Pre-allocated cache with `__setitem__` does NOT work under `fullgraph=True` (first implementation attempt)
- AttnRes adds ~60% step time overhead at this scale — too expensive for wallclock-limited training
- The U-Net skip pattern is surprisingly hard to beat: it's zero-overhead (just tensor addition) and provides a good inductive bias for free

## Verdict

**AttnRes does not beat baseline at this scale/budget.** The U-Net skip connections are the better choice for 9-layer models under a 10-minute wallclock constraint. AttnRes may be worth revisiting if the competition moves to larger models (30+ layers) or longer training budgets.
