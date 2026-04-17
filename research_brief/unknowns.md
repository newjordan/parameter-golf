# Unknowns — Things we don't have data on yet

These are the axes we have NOT swept at the current champion (`L=8, MLP=2, CHAOS_DEPTH=5, MATRIX_LR=0.035`). Any of these could be worth ≥0.01 BPB.

## Chaos-loop dynamics
- **`SCALAR_LR` (default 0.04)**: this is the learning rate for `α, β, φ` — the three parameters that define the sine-perturbation shape. **Never swept.** Possibly the single most consequential untuned knob because it directly controls whether the chaos loop learns meaningful perturbation dynamics or stays near its randn(3) initialization.
- **`chaos_scalars` initialization**: currently `torch.randn(3)` — unstructured Gaussian. No principled scheme based on the chaos-regime landscape. Unknown whether a SIREN-style init (ω₀ tuning, U(−√(6/n)/ω₀, +√(6/n)/ω₀)) improves convergence or stability.
- **`CHAOS_DEPTH` at champion L=8**: only `{5, 7, 10}` have been measured. `{1, 3, 15, 25}` are running in v15 now, but larger depth regimes (30+) and whether the scaling holds at L=8 are open.
- **Jacobian penalty / spectral normalization on `W`**: DEQ-style literature says this is critical for tied-iteration stability, but we have no regularizer on `W`. Unknown whether adding one would help the deeper chaos regimes.
- **Truncated BPTT through chaos loop**: full BPTT through `CHAOS_DEPTH` iterations has the expected memory cost. Unknown whether truncated BPTT (e.g., gradient only through last `k` iterations) would allow deeper chaos at the same memory budget.
- **Random `CHAOS_DEPTH` during training**: some DEQ recipes sample depth stochastically each step to improve robustness. Untried.
- **Separate optimizer / weight decay on chaos params**: `α, β, φ` are currently in the same Adam group as other scalars. Unknown whether isolating them (lower LR, different β₂, no weight decay) changes outcomes.

## Attention and architecture
- **`QK_GAIN_INIT=1.5`**: default, never swept (v15 is testing 0.5 / 1.0 / 2.0 now). Unknown optimal value and whether the relationship to the chaos-loop regime matters.
- **`NUM_HEADS` / `NUM_KV_HEADS`**: fixed at 8/4 GQA for the entire sweep. Never tested 16/8, 4/2, 4/4, or pure MHA.
- **`head_dim`**: derived as `d_model / NUM_HEADS = 64`. Never swept directly.
- **MLA (Multi-head Latent Attention)**: DeepSeek-V2/V3 style — unknown whether it fits or helps under our 16 MB cap.
- **Attention sinks / first-token register**: unknown whether they matter at this model size.
- **Positional encoding**: currently RoPE. Unknown whether ALiBi, NoPE, or partial RoPE would change anything.

## Optimizer and schedule
- **`BETA2=0.95`**: unusually low for Adam. Standard is 0.98-0.999. v15 is testing 0.97 / 0.99 now. Unknown if `BETA2 → 1.0` improves, or if the chaos loop requires fast momentum.
- **`WARMUP_STEPS=20`** (out of 300): aggressive. v15 is testing 10 / 50. Unknown if slower warmup (100?) changes stability.
- **`WARMDOWN_ITERS=200`**: never swept.
- **`GRAD_CLIP_NORM=0.0`** (disabled): unknown if enabling clipping at e.g. 1.0 helps, especially for deeper chaos regimes.
- **Muon at this scale**: we know Muon works for us, but no ablation vs Shampoo, SOAP, Adam-only, or hybrid.
- **Tied `MATRIX_LR` across all matrix params**: unknown if per-layer or per-kind LR (attn vs mlp vs chaos-W) would help.

## Quantization and compression
- **Why does naive int6 destroy weights here?** (+0.2373 BPB on whale). Unknown whether the weight distribution is uniform enough for int8 throughout, or whether per-channel scales fix it, or whether GQKV needs different handling.
- **GPTQ calibration failure mode**: whale log shows `calibrated 0 layers in 3.7s`. Unknown whether the calibration skipped because layer module types don't match GPTQ's expected nn.Linear pattern (custom megakernel blocks).
- **EMA hurt at whale**: post-EMA val_bpb increased by +0.0168. Unknown why — EMA decay, EMA start step, or interaction with our schedule.
- **Artifact headroom (4.83 MB on whale, 2.25 MB on vortex)**: unknown what the right use of it is — wider model at int8, GPTQ with larger group sizes, or dictionary-based weight sharing.
- **Optimal compression algorithm**: `zlib` vs `brotli` vs `zstd` — which gives best size/latency tradeoff for these weight tensors?

## Regularization
- **`FIRST_LAYER_NOISE`**: the model has a hook for input noise, default off. v15 is testing `FIRST_LAYER_NOISE=1, SIGMA=0.01`. Unknown what sigma value is optimal, or whether dropout/stochastic depth is preferable.
- **Weight decay**: `MUON_WD` — not tuned at champion.
- **Label smoothing**: not used. Unknown if it helps at this scale.

## Data
- **Tokenizer size**: 8192 BPE. Unknown if 4096, 16384, or a byte-level tokenizer is better under the 16 MB cap (embedding table scales with vocab).
- **Sequence length**: current `SEQ_LEN` — never swept in recent passes.
- **Batch composition**: `coprime shards` loader with `shards_per_batch=1`, `batch_stride=47`, `hold_steps=64`. Unknown if different shard-cycling strategies affect convergence.

## Fundamental questions
- **Is the chaos loop the right primitive at all?** Or is NUM_LAYERS-scaling evidence that we should go back to a vanilla transformer block and use our megakernel speed for bigger models?
- **Is tying `(W, α, β, φ)` across CHAOS_DEPTH iterations correct?** Universal Transformers tie params, but DEQ lets them untie implicitly through implicit differentiation. No data on untied chaos (would blow the param budget unless CHAOS_DEPTH is small).
- **What does the chaos loop buy us that a wider tanh-only MLP doesn't?** No ablation removing the sine term has been run at champion.
- **If we ran for longer than 10 min**, does the ranking invert? i.e. do deeper chaos regimes need more training to shine, and will lose at 10 min but win at 1 hour?
