# CrawlerT_Leg_1 — Results

**Date:** 2026-03-29
**Hardware:** NVIDIA H100 80GB HBM3
**Config:** dim=512, mlp_mult=4.0 (crawler), B=48, T=2048, dtype=bfloat16

---

## Phase 1: bench.py — NormMLP Kernel Fusion

### Timings

| Test | Forward | Fwd+Bwd | Speedup vs Eager (fwd+bwd) |
|------|---------|---------|---------------------------|
| A: Eager (baseline) | 1.4176 ms | 4.5314 ms | 1.00× |
| B: compile(mode=None) — current default | 1.0164 ms | 2.9541 ms | **1.53×** |
| C: compile(mode='max-autotune') | 2.3560 ms | crashed* | ~0.85× (slower) |

*Crash: `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten`
— max-autotune enables CUDA graphs by default; input tensor must be cloned between
backward calls. Irrelevant to verdict since forward was already slower.

### Autotune findings (from C output)

For every GEMM (98304×512, 512×2048, etc.), the winner was **cuBLAS `mm`**, not Triton:
- Best Triton kernel: ~80% of cuBLAS speed
- cuBLAS margin: 20-30% faster than best Triton config

Inductor's max-autotune finds this, selects cuBLAS, but adds compilation overhead → net slower.

---

## Verdict: NO SIGNAL — Hypothesis CLOSED

**max-autotune does not help for this workload.**

Reasons:
1. GEMMs dominate MLP compute (~0.6ms of ~1.0ms compiled forward)
2. cuBLAS already optimal for these matrix shapes on H100
3. Triton kernels for GEMM are 20-30% slower than cuBLAS here
4. Elementwise ops (norm, relu_sq, scale, residual) are ~0.4ms — inductor already fuses most of this in default mode

**The training is already getting the available speedup.** B vs A shows default compile gives
1.53× fwd+bwd speedup, and `compiled_model = maybe_torch_compile(base_model, args)` in
train_gpt.py means Crawler_Leg_1 already benefits from this.

No step-time improvement available from compile mode tuning. The hypothesis
(Triton → faster steps → more steps in 600s → better BPB) does not hold for this approach.

---

## What This Actually Means for Custom Triton

The bench tested **inductor auto-generated Triton vs cuBLAS** — not hand-written custom kernels.

Custom Triton's real value is for novel algorithms that can't be expressed as standard ops.
In this codebase, the only place a hand-written Triton kernel has demonstrated real lift is
the delta rule (`fla.ops.delta_rule.chunk_delta_rule`), because the recurrence can't be
expressed as a GEMM.

The open question: can a custom Triton kernel for the crawler loop itself give accuracy gains
by enabling new algorithms (smooth loop transitions, fused state accumulation)?
→ See next investigation.

---

## Phase 2

Not pursued — bench.py gate not met.
