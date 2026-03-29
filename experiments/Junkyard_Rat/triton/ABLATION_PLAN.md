# Triton Ablation Plan

Date: 2026-03-29

## Baselines

### `JR-01`
- loader: `coprime`
- kernel: `eager`
- reference sliding BPB: `1.11056240`
- reference step time: `~91.00ms`

### `JR-02`
- loader: `coprime`
- kernel: `triton_act`
- goal: determine whether kernel path is alive

## Stage 0: Bench Sanity

1. `bench_triton.py`
   - purpose: fast signal on the exact MLP path
   - status: proxy only, not final truth

2. Full run with `triton_act`
   - purpose: real answer on system behavior

Decision rule:
- if full run is obviously slower and worse, archive immediately
- if it is neutral or mixed, continue tuning

## Stage 1: Kernel Path A/B

### `TR-01`
- runner: `run_jr02_triton_act.sh`
- variable: `MLP_KERNEL_MODE=triton_act`
- fixed: loader winner, rest of stack unchanged

Primary metrics:
- step time
- `post_ema`
- `final_sliding_window_exact`

## Stage 2: Kernel-Adjacent Runtime Tuning

These stay code-light and keep the same kernel.

### `TR-02`
- variable: `COMPILE_FULLGRAPH=0`
- reason: relax graph capture pressure around the kernel path

### `TR-03`
- variable: `COMPILE_MODE=max-autotune`
- reason: let Inductor tune around the real kernel-backed path, not replace it

### `TR-04`
- variable: `TORCHDYNAMO_SUPPRESS_ERRORS=1`
- reason: only if graph instability appears; not a quality optimization

Decision rule:
- only keep a runtime tweak if it helps either throughput or final BPB

## Stage 3: Numerics Compensation

Only do this if `TR-01` is alive but not clearly winning.

Priority surfaces in code:

1. `mlp_scale`
   - location: `Block.mlp_scale`
   - hypothesis: kernel path may change optimal MLP branch amplitude

2. `resid_mix`
   - location: `Block.resid_mix`
   - hypothesis: kernel-shifted MLP numerics may change the best x/x0 blend

3. `attn_scale`
   - location: `Block.attn_scale`
   - hypothesis: if MLP branch strengthens, attention branch balance may need retuning

4. `ln_scale_factor`
   - location: layerwise norm scaling in `Block`
   - hypothesis: deeper layers may react differently under kernel-altered MLP math

Decision rule:
- one compensation change at a time
- no kitchen-sink "fix numerics everywhere" patch

## Stage 4: Real Fusion

If `TR-01` or tuned descendants stay alive:

Target:
- fuse more of the MLP dataflow
- ideally toward `linear -> leaky_relu -> square -> linear`

Constraint:
- must be shaped for banked weights and the real `48 x 2048 x 512/1536` workload

## Commands

```bash
python experiments/Junkyard_Rat/bench_triton.py
```

```bash
bash experiments/Junkyard_Rat/triton/run_jr02_triton_act.sh
```

```bash
COMPILE_FULLGRAPH=0 bash experiments/Junkyard_Rat/triton/run_jr02_triton_act.sh
```

```bash
COMPILE_MODE=max-autotune COMPILE_FULLGRAPH=0 bash experiments/Junkyard_Rat/triton/run_jr02_triton_act.sh
```
