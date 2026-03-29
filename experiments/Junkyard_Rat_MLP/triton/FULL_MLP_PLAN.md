# Full MLP Fusion Plan

Date: 2026-03-29

## Mission

Turn the current `JR-01` winner into a dedicated full-MLP fusion branch.

This is not the old activation-only path.

Target:
- fuse the MLP body
- measure real end-to-end impact
- only tune numerics after the fused branch is stable

## Current Boundary

Keep outside the first fused kernel:
- `mlp_norm`
- `ln_scale_factor`
- `mlp_scale`
- residual add

Fuse inside:
1. `F.linear(h, up_w)`
2. `leaky_relu(0.5)`
3. `square`
4. `F.linear(act, down_w)`

## Why This Boundary

- It isolates the real MLP dataflow win.
- It avoids mixing fusion work with residual math and norm math in the first pass.
- It keeps attribution cleaner if the branch wins or loses.

## Real Shapes

Optimize for the real per-GPU shape:
- tokens: `48 x 2048 = 98304`
- model dim: `512`
- hidden dim: `1536`

So the hot GEMMs are effectively:
- `98304 x 512` by `512 x 1536`
- `98304 x 1536` by `1536 x 512`

## Implementation Ladder

### `MLP-00`
- branch scaffold only
- no fused kernel yet

### `MLP-01`
- fused forward microbench
- correctness vs eager body

### `MLP-02`
- training integration
- eager backward acceptable if needed for first truth pass

### `MLP-03`
- real backward optimization
- full `600s` comparison against `JR-01`

## Success Rule

Keep the branch alive only if it gives one of:
- lower step time at similar quality
- similar step time with better final BPB
- modest step-time regression with clearly better final BPB

If it gives none of those, archive it.
