# Knowns — VortexHelix Megakernel Project

## Competition context
- **Event**: OpenAI Model Craft Challenge.
- **Hardware**: scoring runs on 8×H100.
- **Artifact cap**: 16 MB after int8 + zlib/brotli compression. Code size counted separately.
- **Metric**: validation bits-per-byte (BPB) on FineWeb val set. **Lower is better.**
- **Scored training wallclock cap**: ~10 minutes. Whatever BPB the model hits inside that window is the score. No "train longer" fallback.
- **Implication**: all optimization is "val_bpb reached in 10 min on 8×H100", not "val_bpb at convergence".

## Current champion
- **Config name**: `lr035_8d_m2`. **val_bpb = 1.9012** at 300 steps × 1×H100. (Previous champ `x_8d_m2` = 1.9039 at the same protocol.)
- **Model config**:
  - `NUM_LAYERS=8`, `MLP_MULT=2`, `ADD_MLP=1`
  - Attention: GQA, `NUM_HEADS=8`, `NUM_KV_HEADS=4`, `head_dim=64`
  - Vocab: `VOCAB_SIZE=8192`, SentencePiece BPE (`fineweb_8192_bpe.model`)
  - Tied input/output embeddings
- **Optimizer**:
  - Muon for matrix params (`MATRIX_LR=0.035`, `MUON_MOMENTUM=0.95`)
  - Adam for embed/head/scalar params (`SCALAR_LR=0.04` default — never swept at champion)
  - `WARMUP_STEPS=20`, `WARMDOWN_ITERS=200`, `ITERATIONS=300` (for sweep protocol)
- **Kernel config** (from v12 sweep):
  - `VORTEX_BLOCK_SIZE=128`, `FWD_NUM_WARPS=8`, `FWD_NUM_STAGES=3`
  - bwd chaos: warps=8 stages=1; bwd attn: warps=8 stages=3; bwd dkv: warps=8 stages=1
- **Artifact size** in champion config: ~13.75 MB / 16 MB (≈2.25 MB headroom).

## VortexHelix architecture (the novelty)
- Training is end-to-end through **one fused Triton megakernel** (`kernels/vortex_fused.py`) that replaces the standard transformer block.
- Inside each layer, a **chaos loop** iterates a tied-parameter recurrence:
  ```
  b_{i+1} = tanh(W · b_i) + α · sin(β · b_i + φ)
  ```
  for `CHAOS_DEPTH` iterations, with **the same (W, α, β, φ) across all iterations**.
- `(α, β, φ)` are three scalar parameters, initialized via `torch.randn(3)` (Gaussian). This init is **unvalidated** — it is known that ~30% of random seeds land in the chaotic-edge regime.
- The output of the chaos loop is bounded by `1 + |α|` regardless of `CHAOS_DEPTH` (Lipschitz of tanh·sin composition).
- `CHAOS_DEPTH` defaults to 5. **It has never been swept at the champion config until v15** (currently running).
- The chaos loop is structurally a **learnable structured noise injection** stacked on top of a tanh fixed-point iteration — the sine term IS the perturbation. Not bolted-on stochastic noise; learned perturbation with tied params.

## What we have evidence for
- **NUM_LAYERS beats CHAOS_DEPTH as an effective-depth axis** (v13 result, 300-step 1×H100):
  - L=1 × CHAOS∈{10, 20, 40, 80}: val_bpb all ≈ 1.979 (flat, delta=0.0004 = pure noise).
  - L=4 × CHAOS=10: 1.9474. L=3 × CHAOS=10: 1.9563. L=2 × CHAOS=40: 1.9655.
  - "One fat chaotic block" hypothesis (L=1, huge CHAOS) is dead.
- **MLP width helps at L=1** (monotonic): m2=1.9791, m3=1.9772, m4=1.9758. ~0.0017 per mult step.
- **BLOCK_SIZE=128 is optimal** (v12): blk64=1.9141 (worst), blk256 OOMs at shared-memory limit.
- **MUON_MOMENTUM=0.95 sweet spot** (v8): 0.90 worst (1.9063 on its row).
- **Depth knee is 8**: L=10 = 1.9097 (worse than L=8 champion).
- **Whale hybrid leg** (totally different architecture — 11L hybrid GPT, 4×H100, 20 min, 1536 steps, 786K tok/batch): val_bpb **1.1856 raw**, **1.4229 after broken quant**. Not comparable to vortex numbers (different arch, ~10× wallclock, ~12× batch).

## Closed paths (do not resurrect)
- **Ouroboros III**: +0.006 BPB vs 9F baseline. Stacked recurrence signals don't compose.
- **Helix_ab_3**: +0.140 BPB hard fail. Micro signal did not scale.
- **Smokestack** (cadence gate, 2026-04-05): fewer loops = worse. RAPID signals unreliable.
- **v11b/v11c 2000-step validation scripts**: nullified by the 10-min wallclock discovery. Kept on disk but misaligned with scoring rubric.

## Whale deploy-pipeline failure mode (separate problem)
- Whale hit val_bpb 1.1856 raw but **quant roundtrip cost +0.2373 BPB** to land at 1.4229.
- Root cause in log: `gptq:calibrated 0 layers in 3.7s` — GPTQ silently no-op'd, fell back to naive int6 on 4 tensors.
- EMA **hurt** by +0.0168 instead of helping.
- 4.83 MB artifact headroom was left unused.
- Indicates our compression/quant pipeline is underspecified for this model family. Relevant to BOTH whale and vortex tracks — whatever we train, we still need to quantize to ≤16 MB.

## Tooling / infra
- 4×H100 pod (`vast-dealer`) is our iteration machine. 1×H100 per trial × 4 parallel = 4 trials in-flight. 300-step trial ≈ 7 min.
- 8×H100 pod (`vast-whale`) exists for full-simulation runs.
- Repo: `newjordan/parameter-golf`, working branch `TON-E`.
- Key file paths (all on GitHub `TON-E`):
  - `test_vortex_2k.py` — training driver, model definition, Muon optimizer, val loop, quant serialization.
  - `kernels/vortex_fused.py` — the megakernel fwd (chaos loop at lines 153-168).
  - `kernels/vortex_bwd.py` — megakernel bwd.
  - `kernels/vortex_function.py` — autograd wrapper.
