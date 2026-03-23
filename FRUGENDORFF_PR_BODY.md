![frugendorff](https://github.com/user-attachments/assets/b99f0186-994a-43f3-9ba4-c7d1a5c055f7)

## The Frugendorff

A recursive weight-sharing architecture for transformer compression. K unique transformer blocks are looped N times with orthogonal position embeddings, yielding deeper effective networks from fewer stored parameters. The freed parameter budget enables larger MLP expansions that would otherwise exceed the artifact size limit.

**Best result: val_bpb 1.1478** | 15.19 MB | 8xH100 SXM, 600s

## Architecture

| | |
|---|---|
| Unique blocks | 6 (Frugendorff Squared) |
| Loops per block | 2 |
| Effective depth | 12 |
| Dimension | 640 |
| Heads / KV heads | 10 / 5 (GQA) |
| MLP expansion | 4x (hidden 2560) |
| Activation | relu-squared |
| Parameters | 28.2M |

**Key components:** Orthogonal loop position embeddings (QR-initialized) ensure each loop operates in non-interfering subspaces. U-Net skip connections within each loop iteration. SmearGate, BigramHash (8192 buckets), shared value embeddings, XSA on last 2 blocks. Tied embeddings, logit softcap 30.

## Training Pipeline

Muon (matrices) + AdamW (embeddings, scalars) · EMA (decay 0.9985) · Late QAT (int6 STE, triggered at LR scale < 0.15) · Training data replay (2 epochs, last 100 batches) · Self-distillation (EMA teacher, 50 steps) · GPTQ int6 + zstd-22 export

## Experiment History

We explored the Frugendorff design space across **400+ experiments** spanning DGX Spark (local), single-GPU H100/G100, and 8×H100 clusters.

### Phase 1: Proof of Concept (DGX Spark, 300-step proxy)

First validation that weight sharing + wider layers beats unique layers at matched parameter count:

| Config | val_bpb | Δ vs baseline | params | dim |
|--------|--------:|----------:|-------:|----:|
| Baseline (9 unique layers, 512d) | 2.7927 | — | 17.05M | 512 |
| **Fractal only (3×3, 864d)** | **2.5953** | **-0.1975** | 16.57M | 864 |
| Fractal + Gravity (3×3, 864d) | 2.6149 | -0.1779 | 16.57M | 864 |
| Fractal + Gravity + AttnRes (3×3, 864d) | 2.6084 | -0.1843 | 16.58M | 864 |

**Finding:** Weight sharing + wider layers is the dominant effect — **7.1% BPB improvement** with fewer total parameters. Gravity (auxiliary losses on early loops) hurt at low step counts; the model learned to suppress early-loop contributions (weights converged to [0.13, 0.13, 0.70]).

### Phase 2: Automated Architecture Search (DGX Spark, 141 + 227 runs)

**Qwen-guided overnight sweep (141 runs):** Automated search over layer count, loop count, cadence, learning rate, gradient clipping, MLP multiplier.

| Axis | Best Value |
|------|-----------|
| Unique layers × loops | 2×4 |
| Cadence | 3 (F/N/N) |
| Learning rate | 2e-3 |
| Gradient clipping | 5.0 |
| MLP multiplier | 3x |

Winner: **2.3332 BPB** (vs 2.6371 baseline — 12% improvement)

**Extended autoresearch sweep (227 runs):** Deeper exploration of the design space:

| Config | Best BPB | Notes |
|--------|----------|-------|
| 4×2, cadence 4 | **2.155** | Overall best |
| 5×2, cadence 4 | 2.185 | Close second |
| 6×1, cadence 1 | 2.196 | No sharing baseline |
| 4×3, cadence 3 | 2.202 | More loops, diminishing returns |
| 6×2, cadence 3 | 2.236 | Used for Squared config |
| 8×2, cadence 3 | 2.329 | Too many unique blocks at this dim |

**Best cadence per setting:** cadence 4 (2.155) > cadence 3 (2.197) > cadence 1 (2.196) > cadence 2 (2.221)

### Phase 3: Full-Scale H100 Runs

| Run | Config | Sliding BPB | Pre-quant | Post-quant | Artifact | Steps | ms/step |
|-----|--------|------------|-----------|------------|----------|-------|---------|
| **Frugendorff Squared** | **6×2, 640d, MLP 4x** | **1.1478** | **1.1570** | **1.1716** | **15.15 MB** | **4,390** | **137** |
| v2 | 3×4, 960d, MLP 3.0 | 1.2113 | 1.2217 | — | 14.2 MB | 5,738 | 105 |
| v3 | 3×4, 960d, MLP 3.3 | 1.2118 | 1.2210 | — | 14.3 MB | 5,590 | 107 |
| v3+TTT | 3×4, 960d, MLP 3.3 | ~1.1901 peak | 1.2210 | — | 14.3 MB | 5,590 | 107 |
| v1 (2-block) | 2×4, 1024d, MLP 3.0 | 1.2715 | 1.2800 | — | 11.3 MB | 7,625 | 79 |
| v1 (3-block) | 3×4, 896d, MLP 3.0 | 1.2111 | 1.2257 | — | 12.8 MB | 5,933 | 101 |
| v6 hybrid | 6×2, 512d, MLP 4x | — | — | 1.1757 | 10.65 MB | — | — |

### Phase 4: Compression Stress Test (G100, single GPU)

Testing GPTQ quantization survival on SwiGLU Frugendorff variants:

| Config | Loops | Pre-quant BPB | Post-quant BPB | Submission Size |
|--------|-------|---------------|----------------|-----------------|
| 11L, share_start=4 | 3 | 1.3766 | **5.716** | 7.66 MB |
| 11L, share_start=4 | 4 | 1.4058 | **6.313** | 7.14 MB |
| 11L, share_start=4 | 5 | 1.4138 | **6.246** | 7.23 MB |
| 11L, share_start=3 | 4 | 1.4076 | **6.214** | 7.35 MB |
| 11L, share_start=4, bigram 4096 | 4 | 1.4123 | — | 7.09 MB |

**Finding:** GPTQ catastrophically degrades weight-shared models. Quantization error compounds multiplicatively across loop iterations. More loops = worse survival.

## Research Directions

Four active threads exploring the novel properties of recursive weight sharing:

### Thread 1: Cadence Training

**Problem discovered:** Running all loops every step causes a "bandsaw" loss oscillation. Shared weights receive contradictory gradient signals from different loop positions on consecutive steps, causing destructive interference.

**Solution:** Cadence training — alternate fractal steps (all loops, full depth) with normalize steps (single clean pass, no loop repetition). The pattern F/N/N/N gives shared weights "recovery" steps where they train without loop-position conflict.

**Evidence:**
- 227-run automated sweep confirms **cadence 4 (F/N/N/N) is optimal** (2.155 vs 2.197 for cadence 3)
- Normalize steps run ~10x faster than fractal steps — cadence 4 gets ~3x more total training steps per wallclock
- The bandsaw disappears with cadence >= 3

**Status:** Sweep complete. Extended single-GPU validation runs in progress (CADENCE=1..5 with per-step logging to visualize bandsaw elimination).

### Thread 2: TTT Leverage Multiplier

**Insight:** Test-time training on shared weights updates all N loop iterations simultaneously — an N× leverage multiplier per gradient step. A single weight update to a shared block improves the model at every depth where that block appears.

**Evidence:**
- v3+TTT on 3×4 Frugendorff: base 1.2217 → **peak 1.1901** (0.032 BPB improvement, ~3× typical TTT gain)
- Aggressive TTT (lr=1e-4, 3 epochs, drift=0.1): peaked at window 1400, degraded to ~1.205 by window 4600
- Conservative TTT (lr=5e-5, 1 epoch, drift=0.05): no improvement — too gentle, weights barely moved
- **Drift gate mechanism:** Lerp weight updates back toward originals to prevent shared-block drift from destabilizing frozen embeddings

**Status:** Sweet spot identified between aggressive and conservative settings (estimated: epochs=2, lr=8e-5, drift=0.08). Systematic sweep planned.

### Thread 3: Quantization Survival

**Problem:** GPTQ quantization destroys weight-shared models. A rounding error that would be tolerable in one layer becomes catastrophic when that same quantized weight is applied 3-5 times in sequence, with each loop's output feeding the next loop's input through the same noisy weights.

**Evidence:** Pre-quant BPB of 1.37-1.41 explodes to 5.7-6.3 post-quant across all tested configurations (see Phase 4 table above). The non-shared baseline survives GPTQ with minimal degradation (1.12 post-quant).

**Proposed solutions under development:**
1. **Selective precision** — keep the shared block in fp16, only quantize unique layers. Cost: ~2-3 MB for the shared block, but eliminates compounding entirely.
2. **Loop-aware GPTQ calibration** — standard GPTQ calibrates each layer independently. Instead, calibrate by running the full looped forward pass so the quantization grid minimizes error over the composed function, not each layer in isolation.
3. **Per-loop dequant offsets** — store small low-rank correction matrices (~50-100KB) per loop iteration that adjust the shared quantized weights differently at each depth. Shared storage, loop-specific inference.

### Thread 4: Loop-Conditioned N-gram Embeddings (Depth-Aware Hashing)

**Problem:** BigramHash is computed once and added before the loops — every loop iteration receives identical n-gram conditioning. The only differentiator between loops is a small learned `loop_pos` vector. Shared weights lack rich depth-aware context about which loop they're currently serving.

**Proposed approach:** Include the loop index in the n-gram hash:
```
Current:   hash(token[t-1], token[t])                        → same for all loops
Proposed:  hash(token[t-1], token[t], loop_idx)              → different per loop
Extended:  hash(token[t-2], token[t-1], token[t], loop_idx)  → trigram × depth
```

Each loop iteration gets fundamentally different input-level conditioning. The shared weights receive rich, distinct information about their current depth — not just a position marker, but different n-gram context that tells the block whether it's doing shallow pattern matching or deep refinement.

**Hypothesis:** If each loop sees sufficiently distinct conditioning, the gradient conflict (Thread 1) may resolve itself — cadence training becomes unnecessary because the bandsaw is eliminated at the input level.

**Cost:** One embedding table region per loop (~100-200KB per loop). Negligible against the 2-5 MB compression savings from weight sharing.

**Status:** Implementation complete. Test sweep ready (control bigram → loop-aware bigram → loop-aware trigram → trigram without cadence).

## Why This Matters

Weight sharing in transformers is underexplored relative to its potential. The Frugendorff demonstrates that:

1. **Fewer unique parameters can outperform more** when the freed budget is reinvested (MLP 4x gives ~2% relative BPB, enabled only by the parameter savings from sharing)
2. **Shared weights create unique training dynamics** (bandsaw oscillation, cadence training) that don't exist in conventional architectures and require novel solutions
3. **Recursive structure multiplies the effectiveness of test-time adaptation** — a property with implications beyond this competition
4. **Quantization of shared weights is a fundamental open problem** — the compounding error is architectural, not algorithmic, and requires new approaches

The architecture consistently fits in 11-15 MB (vs 16 MB budget), leaving headroom for additional capacity or higher-precision components.

## Compliance

No test-time training on validation data. Training replay and self-distillation operate on training data only. All evaluation follows score-first protocol per issue #402.
