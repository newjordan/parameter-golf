# Research Brief — VortexHelix Megakernel

This folder is a self-contained briefing for external research agents. Read these files in order:

1. **[knowns.md](./knowns.md)** — what we've established with evidence (competition context, current champion, architecture, closed paths).
2. **[unknowns.md](./unknowns.md)** — axes we haven't measured at the current champion.
3. **[goals.md](./goals.md)** — what winning looks like and non-goals.
4. **[research_questions.md](./research_questions.md)** — 10 specific questions we want deep research on, with a response template at the bottom.

## Code references

All file paths below are relative to repo root (`newjordan/parameter-golf`, branch `TON-E`):

- `test_vortex_2k.py` — training driver (model def, Muon optimizer, val loop, quant serialization). 1228 lines. Chaos-scalar init is around line 636.
- `kernels/vortex_fused.py` — the fused Triton megakernel (forward pass with chaos loop). The chaos recurrence lives at lines 153-168.
- `kernels/vortex_bwd.py` — backward pass kernel.
- `kernels/vortex_function.py` — autograd Function wrapper that bridges PyTorch and the Triton kernels.

## Tl;dr for an agent with 30 seconds

- Fused Triton megakernel transformer, ~3M params, 8192 BPE vocab, tied embeddings, GQA 8/4.
- Novel component: tied-iteration chaos loop `b_{i+1} = tanh(W·b_i) + α·sin(β·b_i + φ)` for `CHAOS_DEPTH` iterations per layer.
- Scored on val_bpb after 10 min training on 8×H100, artifact ≤16 MB int8+compress.
- Current best val_bpb = **1.9012** at 300 steps × 1×H100 (sweep protocol; extrapolates to ≈same on 8×H100 × 10 min).
- Biggest unknowns: `SCALAR_LR` (drives α/β/φ training), chaos_scalars init scheme, quantization pipeline (whale leg lost 0.24 BPB to naive int6), and whether the chaos loop is the right primitive at all.
