# H100 Quant A/B Robustness Test

## Labeled Test
- Label(s):
  - `quant_ab_1xH100_screen_20260319_215317` (T1)
  - `quant_ab_h100_20260319_223048` (T2)
- Date: `2026-03-19`
- Hardware: `1x NVIDIA H100 NVL`
- Branch: `experiments/quant-robustness-direction`
- Mode: screening run (not 8-GPU control-equivalent)

## Exact Launch Context
- Runner: `bash scripts/run_quant_ab_h100.sh`
- Override used: `NPROC=1`
- Fixed training config remained baseline-equivalent except quant knobs in treatment rows.

## Parsed Metrics
| run | final_int8_zlib_roundtrip_exact val_bpb | pre-quant val_bpb | quantization gap (post-pre) | total submission size int8+zlib (bytes) | train ms | eval ms | train+eval ms | cap check |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| control (8xH100 record) | 1.22436570 | 1.2172 | +0.00716570 | 15863489 | 600038 | 1401 | 601439 | pass |
| T1 (`INT8_CLIP_PERCENTILE=99.9999`, `INT8_KEEP_FLOAT_MAX_NUMEL=65536`) | 1.34875900 | 1.3474 | +0.00135900 | 12987948 | 600376 | 18010 | 618386 | pass |
| T2 (`INT8_CLIP_PERCENTILE=99.99995`, `INT8_KEEP_FLOAT_MAX_NUMEL=131072`) | 1.34904226 | 1.3477 | +0.00134226 | 12970513 | 600494 | 18322 | 618816 | pass |

## T1 Source Snippets (from run output)
- `step:1159/20000 val_loss:2.2750 val_bpb:1.3474 train_time:600376ms`
- `Total submission size int8+zlib: 12987948 bytes`
- `final_int8_zlib_roundtrip_exact val_loss:2.27732308 val_bpb:1.34875900`

## T2 Source Snippets (from run output)
- `step:1150/20000 val_loss:2.2756 val_bpb:1.3477 train_time:600494ms`
- `Total submission size int8+zlib: 12970513 bytes`
- `final_int8_zlib_roundtrip_exact val_loss:2.27780135 val_bpb:1.34904226`

## Screening Recommendation (1xH100)
- `T1` is better on post-quant BPB (`1.34875900` vs `1.34904226`), while `T2` is slightly smaller on artifact size (`12970513` vs `12987948` bytes).
- For quant-loss-first selection, carry `T1` to the next full 8xH100 confirmation run.

## Notes
- This `1xH100` screening run used `world_size:1` and `train_shards:10` in the pasted log, so score is not directly comparable to the archived `8xH100` control for final A/B acceptance.
- Use this run as stability and quantization-gap screening; make keep/reject decision only after full `NPROC=8` T1/T2 completion.
