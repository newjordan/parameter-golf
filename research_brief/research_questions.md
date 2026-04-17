# Research Questions — Things to dig into

Each question has a **why it matters** (link to a specific knob or unknown), a **what to look for** (concrete deliverable), and **how to know you're done** (signal that the question is closed).

If you find a paper, please link it (arXiv URL or DOI). If you find code, link the GitHub. If you have a recommendation, prefix it with `RECOMMEND:` and include the env var or file change.

---

## Q1 — Tied-iteration recurrence stability (DEQ family)

**Why it matters.** Our chaos loop `b_{i+1} = tanh(W·b_i) + α·sin(β·b_i + φ)` with TIED `(W, α, β, φ)` for `CHAOS_DEPTH` iterations is structurally a Deep Equilibrium (DEQ) cell. v13 showed scaling `CHAOS_DEPTH` at L=1 is FLAT (1.979 across c={10,20,40,80}). Two interpretations: (a) the iteration converges to a fixed point fast and extra iterations do nothing, or (b) the iteration is unstable past a few steps and gradients vanish/explode.

**What to look for.**
- Is the iteration likely converging to a fixed point under our init? What spectral radius of the linearization at fixed points would imply convergence?
- DEQ training tricks (Bai 2019, Bai 2021): Jacobian Frobenius regularization, spectral normalization on `W`, quasi-Newton solvers vs naive iteration. Which apply here, with what parameter setting?
- What's the canonical way to verify "the iteration is doing useful work" empirically — gradient norms across iterations? Activation distance `||b_{i+1} - b_i||`?

**How to know you're done.** Either: (a) a recommended regularizer with specific hyperparameters and an expected BPB delta, OR (b) a diagnostic to add to `kernels/vortex_fused.py` that tells us whether the iteration is converging vs collapsing.

---

## Q2 — SIREN-style init for sine activations

**Why it matters.** Our chaos loop has a learnable `α · sin(β · b + φ)` term. SIREN (Sitzmann 2020) showed that sine activations need a specific init — `U(−√(6/n)/ω₀, +√(6/n)/ω₀)` with `ω₀ = 30` for the first layer — or networks fail to train. Our `α, β, φ` are initialized via `torch.randn(3)` with no consideration of this.

**What to look for.**
- Does SIREN-style init apply to our case, where the sine is INSIDE a tanh-iterated loop rather than the activation function itself?
- What's the equivalent of `ω₀` in our setup — is it `β`?
- Recommended init for `(α, β, φ)` given that the loop runs for `CHAOS_DEPTH` iterations and feeds back into itself.

**How to know you're done.** A recommended `chaos_scalars` init (drop-in replacement for `torch.randn(3)` in `test_vortex_2k.py` around line 636) with stated rationale and any caveats about coupling with `SCALAR_LR`.

---

## Q3 — Universal Transformer / Geiping 2025 recurrent depth

**Why it matters.** Geiping et al. 2025 trained a 3.5B param model with up to 32 iterations of recurrent depth — strongest precedent for our tied-iteration design. Our v13 result (NUM_LAYERS still beats CHAOS_DEPTH) contradicts the "single fat block" prediction. Are we doing the recurrence wrong, or is the precedent inapplicable at our scale?

**What to look for.**
- Geiping 2025 specific training tricks (warmup of recurrent depth, learning rate schedule per iteration, regularization). Are they replicable at 8×H100 / 10-min scale?
- Universal Transformer (Dehghani 2018): adaptive computation time, position+timestep embeddings inside the recurrence. Do they have a small-scale equivalent?
- Is there a known scale below which tied-iteration approaches lose to deeper non-tied stacks? Our champion is ≈3M params — far below Geiping's 3.5B.

**How to know you're done.** Either: (a) a recipe to copy that we haven't tried, OR (b) a clear reason why our scale is below the threshold where this works, with an estimate of the param count we'd need to make tied-iteration competitive.

---

## Q4 — Quantization failure mode at int6

**Why it matters.** Whale leg val_bpb went from 1.1856 (raw) to 1.4229 (post-quant) — a +0.2373 catastrophic loss. GPTQ silently no-op'd ("calibrated 0 layers in 3.7s"). Whatever we train, we need to land in ≤16 MB without losing this much BPB.

**What to look for.**
- Why GPTQ might find "0 layers to calibrate" — is it because our blocks aren't `nn.Linear` instances (we use a custom megakernel autograd Function)?
- For weight tensors with our distribution (Muon-trained, tied embed, GQA), what's the expected int6 quantization error compared to int8?
- Per-channel vs per-tensor scales — which gives us the int6/int8 sweet spot?
- Mixed precision (int8 for embed/head, int6 for MLP, int4 for projections?) — what's the optimal mix for ≤16 MB?
- Compression: zlib vs brotli vs zstd on these tensor types — measured size and BPB-impact.

**How to know you're done.** A specific quantization recipe for our model (bit-width per param-kind, scale strategy, compressor) with predicted artifact size and BPB delta.

---

## Q5 — Why is `BETA2=0.95` so low?

**Why it matters.** Standard Adam uses `BETA2 ∈ [0.98, 0.999]`. Ours is 0.95, never tuned. This is the second-moment EMA decay — affects how Adam normalizes gradients. v15 is sweeping {0.97, 0.99} now, but we don't know WHY 0.95 was chosen or what the right value is for our dynamics.

**What to look for.**
- Is there a known coupling between low `BETA2` and recurrent / iterative architectures? Some lit suggests low `BETA2` helps with non-stationary gradients.
- Does Muon have a sensible analogue (it's primarily momentum-based)? Should our Adam-side `BETA2` be even higher to compensate for Muon's noisier matrix-side updates?

**How to know you're done.** A recommended `BETA2` for our setup with rationale, or evidence that 0.95 is correct and shouldn't move.

---

## Q6 — EMA hurt at whale (+0.0168 BPB)

**Why it matters.** Whale's post-EMA val_bpb went UP, not down. This is either a bug, a misconfigured EMA decay, or evidence that EMA is wrong for our setup. We don't currently use EMA in vortex sweeps but probably should at the 8×H100 production level.

**What to look for.**
- Common EMA failure modes: wrong decay rate (too slow / too fast), starting too early (before model has stabilized), interaction with WSD-style decay schedules.
- Whale uses EMA + late-QAT + GPTQ — could the EMA weights be incompatible with the QAT weights at quant time?
- What's the right EMA recipe for ≤10 min training (300-1500 steps)?

**How to know you're done.** A specific EMA configuration (decay, start_step, apply_at) with a hypothesis for why whale's EMA hurt and whether it would hurt or help vortex.

---

## Q7 — Muon at small scale + tied embeddings

**Why it matters.** Muon (Newton-Schulz orthogonalization on momentum) is the matrix-side optimizer that's been winning for us. But it's typically validated at 100M+ params; ours is ~3M. Tied embeddings (input = output) are also unusual at this scale.

**What to look for.**
- Known failure modes of Muon at <10M params.
- Whether the tied-embed gradient (sum of input-side and output-side gradients before Muon ortho) breaks Muon's premise.
- Whether `MATRIX_LR=0.035` is reasonable for our scale or if there's a known scaling rule.

**How to know you're done.** Either: (a) confirmation Muon is fine at our scale with a recommended LR, OR (b) flagging a known failure mode with a fix.

---

## Q8 — Tokenizer choice and embedding budget

**Why it matters.** We use a SentencePiece BPE with 8192 vocab. Embedding table = `vocab × d_model` = ~half of our parameter budget. Larger vocab → bigger embed → less budget for the rest of the model. Smaller vocab → longer sequences for the same content → more compute per token.

**What to look for.**
- For ≤16 MB int8 artifacts on FineWeb-style English data, what's the empirically optimal vocab size?
- Byte-level tokenizers (no learned vocab) — do they fit better under the cap or hurt val_bpb?
- Tied embeddings already shared input/output — anything else to reclaim from the tokenizer side?

**How to know you're done.** Recommended vocab size + tokenizer family with predicted impact on val_bpb and artifact size.

---

## Q9 — H100 throughput ceiling for fused Triton megakernels

**Why it matters.** Our model is one fused Triton kernel per transformer block. H100 has well-known peak tok/sec ceilings depending on kernel design (occupancy, register pressure, shared mem usage). We don't know what fraction of peak we're hitting.

**What to look for.**
- Profiling tools/methodology to measure fraction-of-peak on H100 for a fused Triton kernel like ours.
- Common bottlenecks: TMA, async copies, HBM bandwidth, shared-memory pressure, warp divergence.
- Specific things to try in `kernels/vortex_fused.py` to push throughput (specific lines / patterns to refactor).

**How to know you're done.** A profiling recipe + a list of concrete kernel-level changes to test, ranked by expected throughput delta.

---

## Q10 — First-layer noise / regularization in small-model regime

**Why it matters.** v15 is testing `FIRST_LAYER_NOISE=1, SIGMA=0.01`. Unknown if this matters at 3M params / 300 steps / 64K tok-per-batch.

**What to look for.**
- Known regularizers for small-model / short-training regimes that move val_bpb by ≥0.005.
- Stochastic depth, dropout, label smoothing, MixUp/CutMix-style augmentation on token sequences — applicable here?
- Specifically input-noise regimes: is `σ=0.01` reasonable for normalized token embeddings, or should it be much larger / smaller?

**How to know you're done.** Recommended regularization scheme with hyperparameters and predicted BPB delta.

---

## Format for your response
For each question you address, please use this template:

```
## Q<N> — <one-line restatement>

**Findings.**
<2-5 paragraphs of synthesis, citing papers and code with links>

**RECOMMEND.**
- <specific env var or file change>
- <expected BPB delta or "unknown but worth a 7-min trial">
- <how we'd know we're wrong>

**Priority.** <one of: HIGH (test in next 24h) / MEDIUM (next week) / LOW (interesting but not blocking)>
```

You don't need to answer every question. Prioritize the ones where you find strong, citable evidence. A short, high-signal answer to two questions beats a vague answer to all ten.
