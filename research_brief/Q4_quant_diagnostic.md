# Q4 — Quantization Diagnostic (vortex pipeline)

**Status**: investigation, not a fix. Verify magnitude before investing.

## 1. Current vortex quant pipeline — what is actually happening

Training driver: `/home/frosty40/sota_crawler/test_vortex_2k.py` (1239 lines). Quant code
is self-contained in this file.

### Scheme
- **Format tag**: `"int8_clean_per_row_v1"` (line 391).
- **Quantizer**: naive symmetric int8, per-tensor or per-row, NO calibration, NO GPTQ.
- **Container**: python dict → `torch.save` → `zlib.compress(level=9)` → `final_model.int8.ptz`
  (lines 1190–1197). No brotli. No zstd.
- **Roundtrip validation**: reloads the blob, dequantizes, runs `eval_val` on the
  val split, logs `final_int8_zlib_roundtrip_exact val_loss:.. val_bpb:..`
  (line 1232).

### Exact bit-width per tensor class (see `quantize_state_dict_int8` lines 344–401)
- **2-D float tensors, numel > 65 536**: per-row int8 along dim 0, per-row fp16
  scale. Applies to: `tok_emb.weight` (8192×512 = 4.2M), every
  `attn.c_q/c_k/c_v/proj.weight`, every `mlp.fc.weight`, `mlp.proj.weight`,
  `Block.proj.weight`.
- **1-D / small float tensors ≤ 65 536 elements**: **passthrough as fp16** (not
  quantized at all) — lines 375–379 via `keep_float_tensor`. So
  `q_gain`, `attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights`,
  `chaos_scalars`, `mixer_gate`, `chaos_perturb` survive at fp16.
- **Control-pattern tensors** (`INT8_KEEP_FLOAT_FP32_NAME_PATTERNS`,
  line 298): passthrough as **fp32**, not fp16. Names matching
  `attn_scale,mlp_scale,resid_mix,q_gain,skip_weight,...`.
- **Non-float tensors**: bitwise passthrough.

### Per-channel or per-tensor?
- **2-D tensors**: per-row (dim-0) — so per output-channel for Q/K/V/FC and
  per-input-channel-index for proj. This IS what the research agent asked
  for on "per-channel int8 for GQKV". We already have it. Line 328–336.
- **1-D / tiny**: per-tensor scale (line 339–342).
- **Clip percentile**: 99.99984 (line 309) — symmetric, round-to-nearest.

### Artifact size
- Champion config (NUM_LAYERS=8, MLP_MULT=2, ADD_MLP=1, VOCAB=8192, d=512)
  ≈ 21 M params. Naive int8 payload ≈ 21 MB. After zlib-9: **~13.75 MB**
  (`knowns.md` line 25). Compression ratio 1.53×. `2.25 MB` headroom.

### Does vortex use GPTQ?
- **No.** GPTQ appears nowhere in `test_vortex_2k.py`. The failed WHALE leg's
  `gptq:calibrated 0 layers` log line comes from the separate
  `crawler/TON-E/artifacts/seed_4/train_gpt.py` (a 2865-line driver, vocab 8192,
  model_params = 26 270 292, 103 MB pre-quant artifact). That IS a different
  file with a different quant mode (`int6`/`int8_flat`, `zstd`). The
  research agent is extrapolating from that leg; it does not describe vortex.

### Is there currently any BPB roundtrip loss for vortex?
- **Unknown from existing logs.** The vortex training driver DOES log both
  pre-quant val_bpb (from the validation pass that runs on the final step,
  lines 1083–1108) and post-quant `final_int8_zlib_roundtrip_exact` (line 1232).
  But vortex sweep harnesses parse only the post-quant number — no
  side-by-side delta has been tabulated. **This is the missing measurement.**

## 2. Does the research agent's premise apply?

| Agent's claim                          | Vortex reality                                     |
|----------------------------------------|----------------------------------------------------|
| GPTQ silently no-op'd                  | We never invoke GPTQ on vortex.                    |
| Naive int6 cost +0.2373 BPB            | Vortex uses int8, not int6. That number is WHALE.  |
| Per-channel int8 for GQKV              | **Already implemented** (per-row, line 328–336).   |
| Int6 for MLP, brotli-11, group-size-128| We use int8 everywhere qualifying + zlib-9.        |

**Conclusion**: the recommendation is aimed at a different pipeline. We already
apply per-channel int8 on the large weights. The only overlap: we have not
measured the quant-induced BPB delta for vortex, so we cannot confirm whether
int8 is costing anything here.

## 3. Minimal diagnostic patch (do NOT apply during scored runs — add for one
offline measurement, revert before next champion-class sweep)

Goal: in the SAME trial, log pre-quant val_bpb and post-quant val_bpb back-to-back
on the identical val split, so delta = quant roundtrip loss is a single number.

Already-measured pre-quant BPB: the main loop ALREADY validates at `last_step`
(lines 1083–1105) and logs the line
`step:{step}/{args.iterations} val_loss:.. val_bpb:..`. So the pre-quant number
IS printed but it's the same eval call as the `DIAGNOSTIC` intermediate metric.

### Patch — insert a dedicated pre-quant eval tagged for grep parity

Location: `/home/frosty40/sota_crawler/test_vortex_2k.py`

**After line 1188** (just after `Total submission size` log and BEFORE
`quantize_state_dict_int8(...)` on line 1189), insert:

```python
    # --- Q4 DIAGNOSTIC: pre-quant val_bpb measured on the exact same eval
    # path that final_int8_zlib_roundtrip uses. Enables direct delta = quant loss.
    if os.environ.get("QUANT_DELTA_DIAG", "0") == "1":
        torch.cuda.synchronize()
        _t_raw = time.perf_counter()
        _raw_val_loss, _raw_val_bpb = eval_val(
            args, model, rank, world_size, device, grad_accum_steps,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        log0(
            f"final_prequant_raw_exact val_loss:{_raw_val_loss:.8f} "
            f"val_bpb:{_raw_val_bpb:.8f} eval_time:{1000.0*(time.perf_counter()-_t_raw):.0f}ms"
        )
```

**Then after line 1232** (the existing `final_int8_zlib_roundtrip_exact` log),
insert:

```python
    if os.environ.get("QUANT_DELTA_DIAG", "0") == "1":
        log0(
            f"final_quant_delta_bpb:{q_val_bpb - _raw_val_bpb:.8f} "
            f"raw:{_raw_val_bpb:.8f} post:{q_val_bpb:.8f}"
        )
```

Gated on env `QUANT_DELTA_DIAG=1` so it is bit-identical off by default.
Cost: one extra eval pass per trial (~15–30 s at 300-step scale on 1×H100).

### How to run
```bash
QUANT_DELTA_DIAG=1 \
  bash run_v16_siren_init.sh  # or any representative 300-step trial
```
Grep `final_quant_delta_bpb` — one number per trial.

### Decision thresholds
- **delta < 0.005 BPB**: Q4 is a non-issue. Close it. Ignore research agent
  recommendation. Research agent extrapolated from WHALE; whale != vortex.
- **0.005 ≤ delta < 0.01**: marginal. Park. Revisit only if sweep floor
  approaches the delta magnitude.
- **delta ≥ 0.01 BPB**: real. Escalate to the mixed-precision design below.

## 4. Cap-binding analysis

- Current artifact: 13.75 MB / 16 MB cap → 2.25 MB headroom.
- Embedding tensor (tok_emb.weight, 8192×512): 4.2M params.
  - int8 per-row (current): 4.2 MB + 16 kB scales = 4.2 MB payload.
  - fp16 passthrough: 8.4 MB payload (+4.2 MB vs int8). Over headroom.
  - int6 packed: ~3.1 MB payload (−1.1 MB).
  - **int4 packed** (with per-row scale): ~2.1 MB payload (−2.1 MB), frees 2.1 MB.
- Block matrices (per layer: c_q/c_k/c_v/proj/fc/mlp.proj/Block.proj, ~1.76 MB
  per layer × 8 layers = 14.1 M params):
  - int8 per-row (current): 14.1 MB + 14 kB scales.
  - fp16: 28.2 MB. Way over cap.
- **Headroom is NOT binding on the large matrices at int8** — we have 2.25 MB free,
  which is ~1.3 layers' worth of fp16 weight. Not enough to upgrade any single
  block matrix class to fp16 uniformly.
- The ONE lever: drop embedding to int6 or int4, reclaim 1–2 MB, spend on fp16
  for a subset of small critical tensors or tighter int8 scales. This is only
  worth doing if delta > 0.01 BPB.

## 5. IF delta ≥ 0.01 — mixed-precision design (contingent)

Do not implement until the diagnostic delta is measured. Then:

| Tensor class                            | Current       | Proposed          | Rationale                        |
|-----------------------------------------|---------------|-------------------|----------------------------------|
| `tok_emb.weight` (8192,512)             | int8 per-row  | int6 per-row, g=128 row-groups | Embeddings tolerate more quant; frees 1.1 MB |
| `attn.c_q/c_k/c_v.weight` (per-layer)   | int8 per-row  | int8 per-row (unchanged) | Already matches agent recommendation |
| `attn.proj.weight`, `Block.proj.weight` | int8 per-row  | int8 per-row (unchanged) | Per-row already captures output-channel range |
| `mlp.fc.weight` (d→2d)                  | int8 per-row  | int8 per-row (unchanged) | Already correct |
| `mlp.proj.weight` (2d→d)                | int8 per-row  | **fp16** (if delta concentrates here) | Spend reclaimed embedding budget |
| All 1-D / control tensors               | fp16 or fp32  | unchanged         | Already lossless-ish |

Specific sizes, if all enacted:
- Embedding int6: −1.1 MB
- One MLP proj fp16 per layer: +0.26 MB × 8 = +2.1 MB
- Net: +1.0 MB → blows cap. Trade-off: pick ONE of (embedding shrink) OR (mlp proj upgrade), not both.

Do NOT change the serializer to brotli yet — zlib-9 vs brotli-11 is a 5–15 % byte
win on this payload class, which is 0.7–2 MB. Worth trying ONLY if cap becomes
binding after a weight-layout change. Pure compression swap is a second-order concern.

## Recommendation

**Priority**: MEDIUM-LOW. The research agent's #1-killer framing is anchored on
WHALE's +0.2373 BPB, which came from GPTQ silently failing. Vortex does not use
GPTQ and already runs per-row int8. Until we measure the vortex quant delta,
the claim is unsupported for our pipeline.

**Action**: apply the 10-line diagnostic patch, run ONE 300-step trial with
`QUANT_DELTA_DIAG=1`, inspect `final_quant_delta_bpb`. If < 0.005, close Q4.
Otherwise escalate to the mixed-precision design.

## File/line references
- `/home/frosty40/sota_crawler/test_vortex_2k.py` line 290–305: control tensor patterns.
- `/home/frosty40/sota_crawler/test_vortex_2k.py` line 306–310: clip constants, 65 536 passthrough threshold.
- `/home/frosty40/sota_crawler/test_vortex_2k.py` line 323–342: `quantize_float_tensor` (per-row vs per-tensor branch).
- `/home/frosty40/sota_crawler/test_vortex_2k.py` line 344–401: `quantize_state_dict_int8`.
- `/home/frosty40/sota_crawler/test_vortex_2k.py` line 1189–1197: quant + zlib serialize.
- `/home/frosty40/sota_crawler/test_vortex_2k.py` line 1207–1232: roundtrip validation and log lines.
- `/home/frosty40/sota_crawler/research_brief/knowns.md` line 25: 13.75 MB artifact size.
- `/home/frosty40/sota_crawler/research_brief/knowns.md` line 56–61: WHALE GPTQ failure (NOT vortex).
