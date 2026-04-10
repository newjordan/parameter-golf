# Nightcrawler Cubed

Working submission folder for the current crawler SOTA path.

Source snapshot:
- `train_gpt.py` copied from `crawler/2026-04-09_Trapper_Keeper_1/train_gpt.py`
- lock copied from `crawler/2026-04-09_Trapper_Keeper_1/.train_gpt.lock.json`
- local `run.sh` is the same stack, rebased to run from this folder at repo root

Current confirmed 3-seed reference:
- `seed 444`: `1.13541288` at `15,902,698`
- `seed 300`: `1.13853446` at `15,851,974`
- `seed 4`: `1.13536063` at `15,844,157`
- `mean`: `1.13643599`
- source leg: `crawler/2026-04-09_Trapper_Keeper_1/`

Entry points:
- `run_10min.sh`: exact legal 10-minute stack
- `run_4h.sh`: working copy reserved for the future 4-hour variant

Typical usage:
```bash
SEED=444 NPROC_PER_NODE=8 bash nightcrawler_cubed/run_10min.sh
SEED=300 NPROC_PER_NODE=8 bash nightcrawler_cubed/run_10min.sh
SEED=4   NPROC_PER_NODE=8 bash nightcrawler_cubed/run_10min.sh
```
