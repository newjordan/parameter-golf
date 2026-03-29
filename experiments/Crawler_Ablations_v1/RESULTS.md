# Crawler Ablations v1 — Run Record
Date: 2026-03-29

## Pre-Run Baseline (reference, prior Crawler_Leg_1 run on this pod)

| Metric | Value |
|--------|-------|
| `val_bpb` @ step 6587 | 1.1354 |
| `post_ema val_bpb` | 1.1345 |
| `final_sliding_window_exact` | **1.11099954** |
| step_avg | 91.11ms |
| steps @ 600s | 6587 / 20000 |
| peak memory | 22955 MiB allocated / 23234 MiB reserved |

This is the number to beat on `final_int6_sliding_window_exact`.

---

## Run Attempt 1 — 2026-03-29 ~22:20 UTC
**Command:** `NPROC_PER_NODE=8 NITRUST_ENABLE=0 bash experiments/Crawler_Ablations_v1/run_1gpu.sh`
**Result:** ALL ARMS FAILED
**Root cause:** `CUDA error: invalid device ordinal` — pod has 1 GPU (H100 80GB HBM3), script launched with NPROC_PER_NODE=8.

## Run Attempt 2 — 2026-03-29 ~22:23 UTC
**Command:** `NPROC_PER_NODE=1 NITRUST_ENABLE=0 bash experiments/Crawler_Ablations_v1/run_1gpu.sh`
**Result:** ALL ARMS FAILED
**Root cause:** `OSError: Not found: "./data/tokenizers/fineweb_1024_bpe.model"` — pod not set up. Training data and tokenizer not present.

**Fix:** Run `bash experiments/pod_setup.sh` before re-running.

---

## Pod Spec (2026-03-29)
- GPU: 1× NVIDIA H100 80GB HBM3
- UUID: GPU-8d555ccf-ec65-d1c0-5f30-54572eadcec7
- Host: f7cf0a0a5c85

---

## Pending: Run Attempt 3 (after pod_setup.sh)
**Command:**
```bash
bash experiments/pod_setup.sh
NPROC_PER_NODE=1 NITRUST_ENABLE=0 bash experiments/Crawler_Ablations_v1/run_1gpu.sh
```

Expected: 6 arms × 600s ≈ 60 min total. Results will populate `results/cav1_<ARM>_*/diag.csv`.

### Arms to run
| Arm | Key knob delta | BKD hypothesis |
|-----|---------------|----------------|
| A_baseline | SKIP_GPTQ=0 (GPTQ on), all defaults | — |
| B_loop_aware_gptq | LOOP_AWARE_GPTQ=1 | BKD-10 |
| C_ema_on | SKIP_EMA=0 | BKD-03 |
| D_int8_off | CRAWLER_QUANT_INT8=0 | BKD-09 proxy |
| E_compile_fullgraph | COMPILE_FULLGRAPH=1 | BKD-06 |
| F_gptq_and_ema | LOOP_AWARE_GPTQ=1 + SKIP_EMA=0 | BKD-10+BKD-03 |

### Success gate
Any arm beats **1.11099954** on `final_int6_sliding_window_exact` by ≥ 0.005 → promote to full run.
