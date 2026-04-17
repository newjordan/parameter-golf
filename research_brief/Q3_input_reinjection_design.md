# Q3 — Input reinjection in the chaos loop (design doc)

**Source.** ChatGPT deep research 2026-04-17. Top-ranked recommendation in the deep-research queue.

**Claim.** The current chaos loop
```
b_{i+1} = tanh(W · b_i) + α · sin(β · b_i + φ)
```
is an **autonomous** dynamical map: once `b_0` is set, `b_i` only depends on itself through `W, α, β, φ`. That is a plausible explanation for the repo's own finding that extra `CHAOS_DEPTH` is flat — additional iterations refine the same local state without new input-conditioned computation.

The Geiping 2025 recurrent-depth paper (and Universal Transformer before it) uses an **input-conditioned** recurrent update: the layer input is re-injected at every iteration. The change:
```
b_{i+1} = tanh(W · b_i + U · x_layer) + α · sin(β · b_i + φ)
```

where `x_layer` is the same layer input used to form q/k/v, and `U` is a new tied dim×dim matrix per block.

---

## Where the change lands

### 1. `kernels/vortex_fused.py`

Chaos loop lives at lines 153–168 (current tip). The iteration currently computes:

```python
b_proj = tl.dot(b_bf16, W, out_dtype=tl.float32)      # W·b_i
b_tanh = tanh(b_proj)
b_sin  = tl.sin(beta_val * b_fp32 + phi_val)
b_fp32 = b_tanh + alpha_val * b_sin
```

**Required change.** Add a second matmul `U·x_layer` and fold into the tanh argument:

```python
# NEW: load U and compute Ux once outside the CHAOS_DEPTH loop (x_layer is constant across iterations).
U_ptr = Proj_U + offs_d[:, None] * D + offs_d[None, :]
U     = tl.load(U_ptr).to(tl.bfloat16)
# x_input_bf16 already materialized before the loop (b_fp32 starts from x_layer currently — need to split).
Ux    = tl.dot(x_input_bf16, U, out_dtype=tl.float32)   # [T, D]

for i_step in range(CHAOS_DEPTH):
    b_bf16 = b_fp32.to(tl.bfloat16)
    b_proj = tl.dot(b_bf16, W, out_dtype=tl.float32)
    b_proj = b_proj + Ux                                 # <-- input reinjection
    e_2x   = tl.exp(2.0 * b_proj)
    b_tanh = 1.0 - 2.0 / (e_2x + 1.0)
    b_sin  = tl.sin(beta_val * b_fp32 + phi_val)
    b_fp32 = b_tanh + alpha_val * b_sin
```

**Shared-memory cost.** +1 `[T, D]` tensor (`Ux`) held for the loop duration. At `T=512, D=512, fp32` that is 1 MB/block — within H100 227 KB/block addressable smem if we bf16 the cache (512 KB → still too big). Realistic: rematerialize `Ux` tiles per iteration if smem is tight, or reduce the per-program tile of `T`.

**Register pressure.** Marginal. Adds one persistent tile and one dot product per iteration.

### 2. `kernels/vortex_bwd.py`

Backward must now compute:
- `dU` contribution (similar to `dW` but over `x_layer`).
- `dx_layer` additional path (sum over iterations of `U^T · d(b_proj)_i`).

The existing bwd decomposes chaos gradients across `CHAOS_DEPTH` iterations. Same decomposition applies — U is tied across iterations, so its gradient is a sum over iterations.

**Subagent note.** See `Q1_stability_design.md` for the spectral-norm-on-W slice pattern; U plugs into the same framework — we may want to spectral-norm `[W | U]` jointly if we pursue D3 (contractivity regularizer).

### 3. `test_vortex_2k.py`

`Block.__init__` currently has `self.proj = CastedLinear(dim, dim, bias=True)` which is (re)used as the chaos W. Add:

```python
self.chaos_U = CastedLinear(dim, dim, bias=False)
```

And the Block forward must pass `self.chaos_U.weight` (or `x_layer @ self.chaos_U.weight.T`) into the fused kernel call.

**Parameter budget hit.** `dim × dim = 512 × 512 = 262144` fp32 params per block. At L=8, that is 2.1M new params.
- fp32: +8.4 MB
- bf16: +4.2 MB
- int8 (per-channel): +2.1 MB

Current artifact is 13.75 MB with 2.25 MB headroom against the 16 MB cap. **int8 storage of U exactly eats the headroom.** Reinvestment path is either:
- Shrink `dim_model` from 512 → 448 (≈76% of params, gives back ~2 MB total budget).
- Shrink `NUM_LAYERS` from 8 → 7 (straightforward −12.5%).
- Compress the chaos matrices harder (int6 per-channel with outlier isolation — risky, see Q4).

## Open questions for subagent impl

1. Should `U` and `W` share a layer-norm on `x_layer`? The block normalizes `x_in` → `self.attn_norm(x_in)` for q/k/v. The chaos loop's input `b_0` currently comes from a different path. Decide whether `U · x_layer` uses the pre-norm or post-norm `x`.
2. Does U break tying on chaos params across iterations? U itself is tied (applied once, or its result `Ux` is cached). No new per-iteration weights.
3. Randomizing CHAOS_DEPTH ({3,5,7,9}) only becomes meaningful AFTER this change lands. Schedule as a follow-on.

## Sign of win

If input reinjection works, we should see:
- BPB drop of 0.005–0.020 at L=8 CHAOS=5 (ChatGPT estimate).
- For the first time, CHAOS_DEPTH scaling above 5 should improve BPB rather than flatline (the repo's own v13 result contradicts flatness, and this is the theoretical explanation).

## Sign of loss

If input reinjection doesn't move BPB and costs us the 2.25 MB headroom, retire the direction. That would strongly suggest the chaos primitive is not the right shape regardless of input conditioning, and we should follow the "more layers, simpler blocks" evidence from v13.

## Priority

**HIGH — kernel change.** This is the single highest-EV theoretical win in the whole brief, but it's a 2-hour+ implementation and requires a quant-budget tradeoff. Pair a subagent on the bwd math + kernel shared-memory cost estimate before committing to the full change.
