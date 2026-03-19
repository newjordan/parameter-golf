# Results (H100/X100 Chat-Embedded Runs)
Date: 2026-03-19
Scope: This report summarizes the H100/X100 results referenced in chat (not the local DGX autorun sweep).

## Baseline 10-Min Track Run
- Config: `SP1024`, `9x512`, `KV4`, tied embeddings
- Timed stop: `13780` steps (`~600s` cap)
- Pre-quant metric: `val_bpb = 1.2172`
- Post-quant roundtrip metric (scored): `val_bpb = 1.22436570`
- Artifact size (`int8+zlib + code`): `15,863,489 bytes`

## Unlimited Compute Reference (4h)
- Config family: `SP1024`, `9x512`, `KV4`
- Stop point shown: `329,430` steps
- Pre-quant metric: `val_bpb = 1.1749`
- Post-quant roundtrip metric (scored): `val_bpb = 1.20737944`
- Artifact size (`int8+zlib + code`): `15,810,161 bytes`

## Deltas vs 10-Min Baseline
- Scored improvement: `1.22436570 -> 1.20737944` (`-0.01698626 BPB`, ~`1.39%`)
- Pre-quant improvement: `1.2172 -> 1.1749` (`-0.0423 BPB`, ~`3.47%`)
- Artifact change: `15,863,489 -> 15,810,161` (`-53,328 bytes`)

## Quantization-Robustness Signal
- Baseline quantization gap: `1.2244 - 1.2172 = +0.0072 BPB`
- 4h run quantization gap: `1.2074 - 1.1749 = +0.0325 BPB`
- Interpretation: longer training achieved stronger pre-quant quality, but post-training quantization penalty became larger in absolute BPB terms.
- Directional conclusion: quant robustness is a high-value optimization target.

## Training/Optimizer Context Captured in Chat
- H100 script uses grouped optimizers (Muon + Adam groups), not a single LR:
  - `MATRIX_LR=0.04`
  - `SCALAR_LR=0.04`
  - `TIED_EMBED_LR=0.05`
  - `WARMDOWN_ITERS=1200`
- `WARMUP_STEPS` discussion in chat flagged that script warmup behavior is compile/path priming with state restore, not direct optimization warmup in measured training.

## Failure Note (From Linked Chat)
- Elastic wrapper failure summary showed:
  - `exitcode: 1`
  - `error_file: <N/A>`
  - rank 0 root-cause pointer without full traceback in summary output
- Required debugging follow-up: capture first full traceback from raw run output.
