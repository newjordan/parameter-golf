# Goals — What winning looks like

## Primary objective
**Minimize val_bpb** on FineWeb validation after ≤10 minutes of training wallclock on 8×H100, with the final model artifact ≤16 MB after int8 + (zlib/brotli) compression.

## Concrete targets
- **Beat current best val_bpb = 1.9012** (champion `lr035_8d_m2`) by ≥0.01.
  - Stretch target: val_bpb ≤ 1.88.
  - "Would be a genuine breakthrough" target: val_bpb ≤ 1.85.
- **Must fit ≤16 MB** artifact after compression. Current champion uses ~13.75 MB (≈2.25 MB headroom; whale had 4.83 MB headroom wasted).
- **Must complete a full training + quantize pipeline in ≤10 min on 8×H100** — including EMA application, GPTQ calibration (if used), serialization, and quant roundtrip validation.

## Secondary objectives
- **Throughput ceiling on H100**: the megakernel-first doctrine says until we've maxed out H100 throughput (tok/sec), kernel ablations lead and sizing is co-developed in parallel. Free throughput = bigger model in the same 10 min.
- **Fix the quantization pipeline**: whale showed a 0.24 BPB loss from quant. Whatever model we train, we need to land it in ≤16 MB without double-digit hundredths of BPB lost in the serialize step.
- **Explain WHY the chaos loop helps (or doesn't)**: we have an unusual architecture (tied-param sine perturbation in a tanh fixed-point iteration). Either it's a win we should double down on, or it's dead weight and we should replace it with more layers / wider MLP.

## Non-goals / constraints
- **No "train longer" shortcuts.** Anything that only works past 10 min is off the table.
- **No multi-GPU scaling hacks.** 8×H100 is the scoring rig; we cannot rely on more.
- **No closed paths.** Do not revisit Ouroboros, Helix, Smokestack. Evidence was negative, they're dead.
- **No non-standard data.** FineWeb is the competition corpus. Don't propose curriculum learning on external corpora.
- **No test-time tricks.** The model is evaluated deterministically on val. No beam search, no MC dropout, no ensembling.

## Budget discipline
- Each agent mistake costs real money (~$400 already lost to configuration errors historically). Be precise with numbers. Lower BPB = better. Use "beats" / "worse than", never raw inequalities.
- Don't propose multi-hour runs. The 10-min cap IS the final condition.
- Sweep protocol: 300 steps × 1×H100 per trial ≈ 7 min. 4 GPUs on the dev pod × queue = dense exploration.

## Definition of "done" for a research direction
A research output is actionable if it produces one of:
1. **A concrete ablation to run** (env-var config + hypothesis + expected BPB delta + how we'd know we're wrong).
2. **A kernel-level change** (specific lines in `kernels/vortex_fused.py` / `vortex_bwd.py` to modify, with math justification).
3. **A quantization fix** (specific scheme: per-channel int8, GQKV-aware grouping, dictionary coding, etc. — with an expected BPB impact and artifact size).
4. **A negative result we can retire** — "X was worth investigating, here's why it won't work, stop spending trials on it."

Vague suggestions ("try bigger models", "use better optimizer") are not actionable. Every proposal should name the env var, the kernel file, or the tensor being quantized.
