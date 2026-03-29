# Crawler Backend Optimization Hypothesis Backlog (BKD)

**Experiment:** Crawler_Ablations_v1
**Date:** 2026-03-29
**Status:** Active — 6 arms scheduled

---

## Mission

Map every backend optimization opportunity unique to the crawler's loop-reuse architecture. Focus on what post-processing (SWA, GPTQ) destroys and why.

---

## Architecture Summary

CrawlerGPT: flat encoder layers → shared crawler block (looped N times) → flat decoder layers with skip connections. Per-loop instruction injection via small projections (`inst_dim` ~32). SmearGate + BigramHash + ValueEmbedding in embedding layer. GPTQ quantization at export (int6 for flat layers, int8 option for crawler block).

**Confirmed findings:**
- 85% of crawler advantage comes from WIDTH (fewer unique layers → wider dims at fixed params)
- ~0.007 BPB mid-training lead from shared weight structure is destroyed by SWA + GPTQ
- Quantization catastrophe: 1.38 → 5.7 BPB post-quant at Frug scale; more loops = bigger quant gap
- Root cause of quant gap: activation distribution DRIFT across loop iterations (loop 1 activations differ from loop 3, but one shared quant scale covers all)
- C/N test confirms: iterative refinement provides zero per-step training benefit

---

## Hypothesis Backlog

### Tier 1 — Post-Processing Fix (Highest Impact)

**BKD-01: Per-loop GPTQ calibration**

Collecting activation statistics separately per loop position (loop-1 cal set, loop-2 cal set, etc.) and applying per-position quant scales to the shared block will close the quant gap.

*Why:* GPTQ calibrates quant scales against activation distribution. The shared block sees wildly different distributions at loop 1 vs loop 3. Mixed calibration gives wrong scales for every position.

| Field | Detail |
|---|---|
| New knob needed | `LOOP_GPTQ_CALIBRATION=1` in train_gpt.py |
| Kill gate | No improvement in `final_int6_roundtrip_exact` vs BKD-baseline |
| Status | BLOCKED — needs code in train_gpt.py |

---

**BKD-02: Per-loop activation normalization barrier**

A lightweight rescale-to-unit-variance op between crawler loop iterations will reduce distribution drift, making all loop positions see similar distributions → consistent quant scales → smaller quant gap.

*Why:* Root cause of quant catastrophe is drift. Normalization does not need learned params — just rescale.

| Field | Detail |
|---|---|
| New knob needed | `CRAWLER_LOOP_NORM=1` |
| Kill gate | Step time regression > 5% with no BPB or quant gap improvement |
| Status | BLOCKED — needs code in train_gpt.py |

---

**BKD-03: Loop-decoupled EMA rates (ARM C in this run)**

Using a slower EMA update rate for the shared crawler block (while keeping flat layers at normal rate) will preserve the convergent structure that shared weights learn, preventing SWA from averaging away the ~0.007 BPB mid-training signal.

*Why:* The shared block must generalize across all N loop positions simultaneously. Fast EMA averages in recent checkpoints that were specialized. The measured 0.007 BPB PD mid-training lead getting wiped in SWA is the evidence.

| Field | Detail |
|---|---|
| Testable now (partial) | `SKIP_EMA=0` tests whether EMA on/off affects shared blocks differently than flat layers |
| Full test requires | `EMA_DECAY_CRAWLER` knob |
| Kill gate | `SKIP_EMA=0` arm is worse than `SKIP_EMA=1` arm on `final_int6_sliding_window_exact` |
| Status | PARTIAL — arm C tests EMA on/off interaction; per-component decay needs new knob |

---

### Tier 2 — Compute Scheduling

**BKD-04: Asymmetric activation checkpointing**

Aggressively checkpointing crawler block activations (cheap recompute — same weights, hot in L2) but NOT flat layer activations (expensive — unique cold weights) will reduce peak memory without throughput regression.

*Why:* Standard checkpoint treats all layers uniformly. Crawler loops are cheapest to recompute because shared weights are still resident in cache after the first loop.

| Field | Detail |
|---|---|
| New knob needed | `CRAWLER_RECOMPUTE=1`, `FLAT_RECOMPUTE=0` |
| Status | BLOCKED — needs code |

---

**BKD-05: Fused multi-loop CUDA kernel**

Executing all N crawler loops as a single kernel launch (weights resident in shared memory between iterations) will improve throughput 8–15%.

*Why:* Each loop currently incurs full kernel launch overhead plus L2 cache eviction. Same weights are loaded and evicted N times.

| Field | Detail |
|---|---|
| New requirement | Custom Triton/CUDA kernel — significant engineering |
| Status | BLOCKED — needs kernel implementation |

---

**BKD-06: torch.compile loop unrolling (ARM E in this run)**

Manually unrolling the crawler loop before compile (N sequential identical ops sharing weights) will let torch.compile fuse across loop boundaries and eliminate redundant computations.

*Why:* torch.compile cannot optimize Python `for` loops well. Manual unrolling exposes the static structure.

| Field | Detail |
|---|---|
| Testable now | `COMPILE_FULLGRAPH=1` (existing knob, tests compile behavior on crawler) |
| Kill gate | NaN, runtime error, or step time regression > 10% with no BPB improvement |
| Status | PARTIAL — COMPILE_FULLGRAPH=1 tests the compile path; full loop unrolling needs a code change |

---

### Tier 3 — Instruction Bottleneck

**BKD-07: Batched instruction projection GEMM**

Replacing N sequential `inst_up[i](inst)` calls with a single batched GEMM over all loop instruction matrices will improve throughput because small GEMMs become one tensor-core-friendly GEMM.

*Why:* N separate (dim=600, inst_dim=32) GEMMs are too small for tensor-core efficiency. Batching reaches friendly shapes.

| Field | Detail |
|---|---|
| Status | BLOCKED — needs code in CrawlerGPT instruction path |

---

**BKD-08: Fused proj+up into single low-rank GEMM**

Precomputing `proj @ up_i` products eliminates the sequential inst_dim bottleneck: two GEMMs become one per loop.

*Why:* Especially valuable at small inst_dim (32). At dim=600, inst_dim=32: two 600×32 + 32×600 ops → one 600×600 rank-32 op.

| Field | Detail |
|---|---|
| Status | BLOCKED — needs code |

---

### Tier 4 — Quantization Architecture

**BKD-09: Per-loop quantization scale banks (ARM D indirectly)**

Storing N sets of quantization scales for the shared crawler block (one per loop position) will match scales to actual per-loop activation distributions.

*Why:* Direct fix for the multi-distribution problem. N× scale storage is negligible bytes. ARM D tests `CRAWLER_QUANT_INT8=0` as a proxy — if int6 quant on the crawler block causes less catastrophic failure at loop-matched precision, it validates the distribution mismatch hypothesis.

| Field | Detail |
|---|---|
| Status | PARTIAL — needs new export policy and scale bank infrastructure |

---

**BKD-10: Loop-ordered GPTQ execution (ARM B in this run)**

Running GPTQ in execution order (flat encoder → crawler loop 1 → crawler loop 2 → ... → flat decoder), passing activations from each quantized layer to calibrate the next, will produce better int6 quality than one-shot GPTQ.

*Why:* Current GPTQ treats the model as static layers. Loop 2's activation distribution depends on how loop 1 was quantized. Sequential calibration captures propagated error.

| Field | Detail |
|---|---|
| Testable now | `LOOP_AWARE_GPTQ=1` (existing knob) |
| Kill gate | `final_int6_roundtrip_exact` equal or worse vs ARM A baseline |
| Status | TESTABLE — ARM B |

---

### Tier 5 — Training Dynamics

**BKD-11: Per-loop gradient scaling**

Downweighting early-loop gradients (noisier, block has not refined yet) and upweighting later-loop gradients will reduce noise in shared weight updates.

| Field | Detail |
|---|---|
| Status | BLOCKED — needs gradient hook in training loop |

---

**BKD-12: Skip connection contiguous pre-allocation**

Pre-allocating a single contiguous tensor for all encoder skip connections (instead of a Python list) will reduce decoder-phase cache misses.

| Field | Detail |
|---|---|
| Status | BLOCKED — needs code change in U-Net forward path |

---

### Tier 6 — Inference

**BKD-13: Convergence-gated early loop exit**

Computing `||h_n - h_{n-1}|| / ||h_{n-1}||` after each loop and skipping remaining loops when below threshold will reduce inference compute for converged positions.

| Field | Detail |
|---|---|
| Status | BLOCKED — inference-only change, no training impact |

---

**BKD-14: Approximate KV reuse between loops**

Computing exact KV at loop 1 and applying a low-rank correction for loops 2+ will reduce inference compute by ~(N-1)/N of KV computation.

*Why:* Loop instruction perturbation is small (std=0.01 init). Q/K/V projections at loop N should be close to loop 1 values.

| Field | Detail |
|---|---|
| Status | BLOCKED — inference-only |

---

## Run Protocol

Six arms, single GPU (1×H100), 600s each.

| Arm | BKD | Key delta from baseline |
|---|---|---|
| A_baseline | — | SKIP_GPTQ=0, SKIP_EMA=1, all defaults |
| B_loop_aware_gptq | BKD-10 | LOOP_AWARE_GPTQ=1 |
| C_ema_on | BKD-03 | SKIP_EMA=0 |
| D_int8_off | BKD-09 proxy | CRAWLER_QUANT_INT8=0 |
| E_compile_fullgraph | BKD-06 | COMPILE_FULLGRAPH=1 |
| F_gptq_and_ema | BKD-10 + BKD-03 | LOOP_AWARE_GPTQ=1 + SKIP_EMA=0 |

**Primary metrics:** `final_int6_roundtrip_exact`, `final_int6_sliding_window_exact`

**Secondary metrics:** step time (ms), quant gap (pre-quant val_bpb − post-quant roundtrip)

---

## Success Gates

| Condition | Action |
|---|---|
| Any arm beats ARM A on `final_int6_sliding_window_exact` by >= 0.005 BPB | Promote to full run |
| Step time regression > 10% with no BPB gain | Kill |
| NaN or runtime error | Kill, log blocker |
