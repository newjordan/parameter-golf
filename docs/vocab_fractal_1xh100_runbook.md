# Fractal + Vocab Expansion 1xH100 Runbook

Branch: `experiments/vocab-3lane-fractal-1xh100`

## Goal
Run a viable 1xH100 fractal test for vocab expansion (default lane: `L2_sp1536_dim504`) with minimal manual setup.

## One-Command Pod Run
On a fresh pod with enough free volume (`>=120GB`, prefer `>=150GB`):

```bash
cd /workspace
git clone --depth 1 --branch experiments/vocab-3lane-fractal-1xh100 https://github.com/newjordan/parameter-golf.git
cd /workspace/parameter-golf
python3 -m pip install -r requirements.txt
bash scripts/run_vocab_l2_1xh100.sh
```

This does:
1. Build `sp1536` tokenizer/data (`TOKENIZER_CONFIG=data/tokenizer_specs_sp1536_only.json`)
2. Run `LANES=L2_sp1536_dim504` with fractal enabled on `NPROC=1`

## Cost-Efficient Flow (Recommended)
Do data build on DGX/CPU machine, run training on H100 only.

1. Build artifacts on DGX:
```bash
cd /home/frosty40/parameter-golf-lab
git checkout experiments/vocab-3lane-fractal-1xh100
git pull --ff-only origin experiments/vocab-3lane-fractal-1xh100
TOKENIZER_CONFIG=data/tokenizer_specs_sp1536_only.json TOKENIZER_TRAIN_DOCS=500000 bash scripts/build_vocab3_data.sh
```

2. Copy artifacts to pod:
```bash
rsync -avh --progress -e "ssh -i ~/.ssh/id_ed25519" \
  /home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_1536_bpe.model \
  /home/frosty40/parameter-golf-lab/data/tokenizers/fineweb_1536_bpe.vocab \
  ob8c701lccpvhc-6441170d@ssh.runpod.io:/workspace/parameter-golf/data/tokenizers/

rsync -avh --progress -e "ssh -i ~/.ssh/id_ed25519" \
  /home/frosty40/parameter-golf-lab/data/datasets/fineweb10B_sp1536/ \
  ob8c701lccpvhc-6441170d@ssh.runpod.io:/workspace/parameter-golf/data/datasets/fineweb10B_sp1536/
```

3. Run on pod without rebuilding data:
```bash
cd /workspace/parameter-golf
SKIP_BUILD=1 bash scripts/run_vocab_l2_1xh100.sh
```

## Script Controls
`scripts/run_vocab_l2_1xh100.sh` supports:
- `NPROC` (default `1`)
- `MAX_WALLCLOCK_SECONDS` (default `600`)
- `LANES` (default `L2_sp1536_dim504`)
- `SKIP_BUILD` (`1` to skip tokenizer/data build)
- `RUN_TAG` (output folder tag)

## Expected Outputs
- Training logs:
  - `records/track_non_record_16mb/<RUN_TAG>/L2_sp1536_dim504/train.log`
- Data artifacts:
  - `data/tokenizers/fineweb_1536_bpe.model`
  - `data/tokenizers/fineweb_1536_bpe.vocab`
  - `data/datasets/fineweb10B_sp1536/`

## Known Failure Modes
- `Disk quota exceeded` during data build:
  - free cache/logs or increase pod volume.
- Git write failures (`.git/index.lock`, `unpack-objects` quota errors):
  - clear locks and free disk first; prefer fresh pod if persistent.
