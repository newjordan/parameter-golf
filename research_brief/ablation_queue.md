# Ablation Queue — Tomorrow's Fire List

**Source.** ChatGPT deep research 2026-04-17 (`external_responses/chatgpt_deep_research_01.md`) + repo evidence.
**Current champion.** `champ_qk05` val_bpb **1.89896** (QK_GAIN_INIT=0.5, L=8 MLP=2 CHAOS=5 MATRIX_LR=0.035, all other vortex defaults).
**Baseline to beat.** 1.9012 (old) → 1.89896 (new, post v15). Any new variant must beat **1.89896**.

Tiers: **HIGH** = fire tomorrow AM; **MED** = fire after HIGH drains; **LOW** = park unless HIGH/MED opens new signal; **KERNEL** = needs code change before running.

---

## HIGH — fire tomorrow morning

### A1. Conservative chaos-scalar init (v18)
- **Script.** `run_v18_conservative_siren.sh` (8 trials × 4 GPUs ≈ 15 min).
- **Env.** `CHAOS_SCALAR_MODE=conservative` → α∈U(-0.10, 0.10), β∈U(0.8, 1.2), φ∈U(-π, π). Gate now in `test_vortex_2k.py:~636`.
- **Variants.** 3 seeds at champion config, plus SCALAR_LR ∈ {0.02, 0.04, 0.08}, plus pair with QK_GAIN_INIT=0.5.
- **Why.** v16 empirical + ChatGPT deep research both say β>>1 destabilizes the tied loop. Existing `CHAOS_SIREN_OMEGA0` gate is broken (uses β∈[5,45]). Fresh gate re-tests SIREN with the CORRECT amplitude/frequency regime.
- **Expected delta.** 0.003–0.010 BPB, mostly via seed-variance reduction.
- **How we'd know we're wrong.** If 3 seeds × conservative mode all land within 0.002 of 1.89896 randn baseline, chaos-scalar init is not the lever.

### A2. QK_GAIN_INIT fine sweep (v19)
- **Script.** `run_v19_qk_fine_sweep.sh` (6 trials × 4 GPUs ≈ 12 min).
- **Env.** `QK_GAIN_INIT` ∈ {0.25, 0.35, 0.40, 0.55, 0.60, 0.75} at v18-best chaos-scalar mode.
- **Why.** v15 showed 0.5 > {1.0 default, 2.0}. Never explored below 0.5 or between 0.5 and 1.0.
- **Expected delta.** 0.001–0.005 BPB.
- **How we'd know we're wrong.** If all 6 are within ±0.002 of 1.89896, QK_GAIN plateau — move on.

### A3. All-int8 per-channel quant audit
- **Not a training trial.** Code-path audit of `test_vortex_2k.py` quant serialization.
- **Why.** ChatGPT Q4 finding: GPTQ whale disaster was due to `nn.Linear` detection; vortex path is different. Verify current vortex quant uses all-int8 per-row/per-channel (not naive int6 fallback).
- **Deliverable.** Write expected-vs-actual into `research_brief/Q4_quant_diagnostic.md` (already drafted — extend).
- **Expected delta.** Risk removal (avoid whale-style +0.24 BPB loss).

---

## MED — fire after HIGH drains or in parallel on 4th GPU

### B1. MUON_WD sweep (v21)
- **Script.** `run_v21_muon_wd.sh` (5 trials × 4 GPUs ≈ 10 min).
- **Env.** `MUON_WD` ∈ {0.0, 0.01, 0.02, 0.05, 0.1} at champion.
- **Why.** ChatGPT Q7: Muon scaling paper identifies weight decay as one of the two crucial scaling ingredients. Never swept at vortex scale.
- **Expected delta.** 0.002–0.008 BPB.

### B2. VOCAB_SIZE=4096 + depth reinvestment (v20)
- **Script.** `run_v20_vocab4k.sh` (4–6 trials).
- **Env.** `VOCAB_SIZE=4096` + (L=8 MLP=2) vs (L=10 MLP=2) vs (L=12 MLP=2) vs (L=8 MLP=3).
- **Prereq.** Need 4k BPE tokenizer at `/workspace/Fartmagic/data/tokenizers/fineweb_4096_bpe.model`. If absent, build first (SentencePiece train ~5 min on FineWeb subset).
- **Why.** ChatGPT Q8: 8192 vocab × 512 dim = 4.19M params (half model budget). Dropping to 4k frees ~2.1M params for depth/width.
- **Expected delta.** 0.005–0.020 BPB if reinvestment lands; near-flat if 4k alone.
- **How we'd know we're wrong.** 4k at current shape worse than 8k baseline, AND L=10/L=12 variants don't recover — then vocab is correctly sized already.

### B3. FLN fine-sigma sweep
- **Env.** `FIRST_LAYER_NOISE=1, FLN_SIGMA` ∈ {0.001, 0.003, 0.005}.
- **Why.** ChatGPT Q10: σ=0.01 may already be too large for small embed. v15 tested 0.01 only.
- **Expected delta.** 0–0.004 BPB.

### B4. Separate chaos-scalar optimizer group
- **Requires code change.** Add param-group split in optimizer setup so α, β, φ get their own Adam group.
- **Why.** ChatGPT Q5: chaos scalars are the most non-stationary state; deserve their own BETA2 (keep 0.95) while rest of Adam side can move to 0.97.
- **Expected delta.** 0–0.004 BPB.

---

## LOW — park unless new signal opens

### C1. Late-start EMA (step 180, decay 0.99)
- **Why.** ChatGPT Q6: whale's EMA failure was horizon mismatch, not architecture. Late-start short-horizon EMA is the correct recipe for 300-step sweeps.
- **Deprioritized.** Only valuable for 8×H100 production run, not sweep. Test in production run only.

### C2. Random CHAOS_DEPTH during training
- **Why.** ChatGPT Q3: depth stochasticity is part of the Geiping 2025 recipe.
- **Deprioritized.** Only meaningful AFTER input-reinjection kernel change (D1). Without reinjection, randomized depth is still autonomous-attractor scheduling.

### C3. GRAD_CLIP_NORM=1.0
- **Why.** Never tested at champion. DEQ literature often requires clipping.
- **Expected delta.** Unknown; mostly stability insurance.

### C4. `BETA2=0.97` global
- **Why.** ChatGPT Q5: 0.97 is the only realistic challenger to 0.95.
- **Deprioritized.** v15 already tested 0.97 at old champ; re-test only if B4 (split group) lands unclear.

---

## KERNEL — requires code change; schedule separately

### D1. Input-reinjection in chaos loop (**TOP PRIORITY kernel change**)
- **Change.** `b_{i+1} = tanh(W·b_i + U·x_layer) + α·sin(β·b_i + φ)` — inject per-iteration the block input `x_layer` (same one used to form q/k/v) through a new tied matrix U.
- **Files.** `kernels/vortex_fused.py` (chaos loop ~lines 153-168), `kernels/vortex_bwd.py` (matching backward), `test_vortex_2k.py` (new U parameter per block).
- **Param cost.** +1 dim×dim matrix per block ≈ +4M params at d=512. Budget hit: ~8 MB at fp32, ~4 MB at fp16, ~2 MB at int8 → eats all 2.25 MB artifact headroom. **Would force shrinking elsewhere (L=7 or d_model=448).**
- **Why.** ChatGPT Q3: this is THE missing ingredient from the Geiping 2025 recurrent-depth recipe. Predicted to explain why deeper CHAOS_DEPTH flatlines (autonomous attractor vs input-conditioned iterative refinement).
- **Expected delta.** 0.005–0.020 BPB IF salvageable. Highest-EV lever in the whole brief.
- **Design doc.** `research_brief/Q3_input_reinjection_design.md` (new — drafted in this prep).

### D2. Convergence diagnostic in chaos kernel
- **Change.** Export per-iteration `‖b_{i+1}−b_i‖/(‖b_i‖+ε)` and `cos(b_{i+1}, b_i)` for one debug batch.
- **Files.** `kernels/vortex_fused.py` — add an optional debug tensor output.
- **Why.** ChatGPT Q1: before running more CHAOS_DEPTH experiments, need to know empirically whether the loop is converging, collapsing, or oscillating.
- **Expected delta.** Not a BPB change itself. Informs whether to invest in contractivity regularizer (C5 below) or retire CHAOS_DEPTH>5.

### D3. Spectral-norm / contractivity regularizer on W
- **Change.** Power-iteration estimate of `‖W‖₂` + scalar penalty `max(0, ‖W‖₂ + |αβ| - 0.95)²`.
- **Files.** `test_vortex_2k.py` (loss aux term) — W comes from `Block.proj.weight` slice (see Q1 design doc).
- **Why.** ChatGPT Q1: enables deeper useful recurrence by bounding the Jacobian norm.
- **Expected delta.** 0 at depth 5; 0.005–0.015 BPB if unlocks depth 7–15.
- **Design doc.** `research_brief/Q1_stability_design.md` (already drafted).

### D4. Nsight Compute roofline profile
- **Deliverable.** Achieved occupancy, registers/thread, smem/block, tensor utilization, DRAM throughput, L2 hit rate, top stall reasons for the fused chaos kernel on one training step.
- **Why.** ChatGPT Q9: "fraction of peak" is a guess until measured. Likely 5–20% free throughput.
- **Requires.** Nsight Compute installed on pod + single-rank debug run.

### D5. Triton autotune grid around fused kernel
- **Change.** Add `@triton.autotune` over tile shape × num_warps × num_stages in `kernels/vortex_fused.py`.
- **Why.** ChatGPT Q9: current config is hand-picked, not exhaustive. v17 partial sweep of num_warps/num_stages is a manual version of this.
- **Expected delta.** 5–15% throughput → translates to bigger model in 10-min cap.

---

## Ordering rationale

1. **v18 first** (conservative SIREN) — cheapest test of the highest-confidence rec. Validated by both empirical (v16) and theoretical (ChatGPT DR) evidence.
2. **v19 second** (QK fine sweep) — compounds on v18; may unlock a joint optimum.
3. **v21 third** (MUON_WD) — simple env-var sweep, well-evidenced.
4. **v20 fourth** (4k vocab) — structural bet; biggest upside-per-trial if tokenizer prereq is fast.
5. **D1 (input reinjection)** is the biggest theoretical win but requires kernel changes + param budget reallocation. Schedule as its own morning task with subagent pairing.
6. **D2/D3** unlock deeper CHAOS_DEPTH only if D1 doesn't obsolete them.
7. **D4/D5** are throughput work — megakernel-first doctrine says they run in parallel with BPB work, not sequentially.

## What's closed (don't retest)
- Whale hybrid (CLOSED — 1.4229 post-quant disaster)
- Ouroboros (CLOSED)
- Helix (CLOSED)
- Smokestack (CLOSED)
- `CHAOS_SIREN_OMEGA0` ≥ 10 (CLOSED by v16 — catastrophic, keep the gate only for regression tests)
- SCALAR_LR ∈ {0.01, 0.02, 0.08, 0.16} (CLOSED by v15 — all worse than 0.04)
- BETA2 0.97 global (weakly closed by v15, confirmable after B4)
- VORTEX_BLOCK_SIZE=256 (CLOSED by v12 — OOM)
