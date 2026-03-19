# Parameter Golf Conversation Synthesis
Date: 2026-03-19 (America/Chicago)
Scope: Strategy + execution decisions captured from the working thread, including the late-night failure note from 2026-03-18 11:42 PM.

## 1) What was established
- The official H100 baseline does not use a single LR. It uses Muon + Adam groups with separate rates:
  - `MATRIX_LR=0.04` (Muon, matrix params)
  - `SCALAR_LR=0.04` (Adam, scalar/vector params)
  - `TIED_EMBED_LR=0.05` (Adam, tied embedding)
  - `WARMDOWN_ITERS=1200`
- The local overnight AdamW finding (`lr=1e-3` > `3e-4`) is useful as a signal that defaults can be beat, but the exact LR values do not directly transfer to Muon.

## 2) Critical code correction (important)
- In `train_gpt.py`, `WARMUP_STEPS` is compile/path priming and then model/optimizer state is restored.
- Therefore, `WARMUP_STEPS` is not an optimization warmup schedule in this script.
- Real train-time schedule knobs are:
  - `WARMDOWN_ITERS` via `lr_mul(...)`
  - Muon momentum warmup via `MUON_MOMENTUM_WARMUP_START` and `MUON_MOMENTUM_WARMUP_STEPS`

## 3) Vocab strategy verdict
- Vocab changes are valid for the competition, but tokenizer edits are examined more carefully.
- BPB math supports tokenizer work (`bits_per_token * tokens_per_byte`), so vocab can be a strong lever.
- This is not fully "free":
  - tied embedding is also logits projection, so `VOCAB_SIZE` increases logits compute.
  - artifact headroom is limited in the baseline.

## 4) Size/headroom context used for planning
- Baseline record:
  - `val_bpb=1.22436570`
  - total artifact `15,863,489` bytes
  - cap is `16,000,000` bytes
- Practical implication: large vocab jumps without shape adjustments are risky for the cap.

## 5) Agreed 3-lane vocab experiment
- Lane A: `sp1280 + MODEL_DIM=504`
- Lane B: `sp1536 + MODEL_DIM=504`
- Lane C: `sp2048 + MODEL_DIM=496`
- Purpose: test higher-vocab effect while keeping model size near cap-safe range.

## 6) Branch + assets created
- Branch: `experiments/vocab-3lane`
- Added:
  - `scripts/vocab_3lane_sweep.sh`
  - `scripts/build_vocab3_data.sh`
  - `data/tokenizer_specs_vocab3.json`

## 7) Current blocker and resolution
- Cached dataset manifest does not include `sp1280/sp1536/sp2048`.
- So the sweep cannot pull these variants directly from cached exports.
- Required first step: retokenize/export data locally:
  - `bash scripts/build_vocab3_data.sh`
- Then run lanes:
  - `bash scripts/vocab_3lane_sweep.sh`

## 8) Late-night failure note (from pasted log)
- Timestamp in note: 2026-03-19 04:40:35 UTC-equivalent log context; user-referenced as "Yesterday at 11:42 PM".
- PyTorch elastic report shows:
  - `exitcode 1`
  - `error_file: <N/A>`
  - no direct traceback emitted in that summary.
- Action already suggested in-thread:
  - run with `TORCHDYNAMO_VERBOSE=1` and capture full output to reveal the first real exception.

## 9) Recommended immediate sequence
1. `cd /home/frosty40/parameter-golf-lab`
2. `git checkout experiments/vocab-3lane`
3. `bash scripts/build_vocab3_data.sh`
4. `bash scripts/vocab_3lane_sweep.sh`
5. If any run fails, capture full first traceback (not only elastic wrapper summary) and pin the root cause.
