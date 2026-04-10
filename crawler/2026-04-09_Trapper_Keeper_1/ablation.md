# Ablation: Trapper_Keeper_1

Date: 2026-04-09
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## Production gate (8xH100, 600s, seed=444)

Status: [x] QUALITY PASS — ARTIFACT OVER CAP

Command: `SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-09_Trapper_Keeper_1/run.sh`

| Metric | Value |
|--------|-------|
| model_params | 31,777,372 |
| raw_bpb | 1.1522 |
| int6_sw_bpb | 1.13526829 |
| step_avg_ms | 157.47 |
| steps | 3,811 |
| bytes_total | 17,948,983 (OVER 16MB cap by 1.95MB) |
| artifact_legal | NO |

Gate: beat 1.13867894 int6_sw_bpb — YES (-0.00341)
Gate: artifact <= 16,000,000 bytes — NO (17,948,983 with zstd)

Note: pod did not have FA3. Production step time with FA3 would be ~110ms → ~5,450 steps.
This run got 3,811 steps at 157ms. Quality would likely be BETTER with FA3+more steps.

## Blocker: artifact size

zstd compression: 17,948,983 bytes (1.95MB over)
Brotli patch committed but not yet tested on this artifact.
Expected brotli savings: 10-15% → ~15.3-16.1MB. Needs verification.

## Recovered disconnected run (8xH100, 600s, seed=444, 7F+3C)

Status: [x] QUALITY PASS — NEAR-MISS SIZE FAIL

Command: `SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-09_Trapper_Keeper_1/run_7f3c_brotli.sh`

Raw recovered log:
`crawler/2026-04-09_Trapper_Keeper_1/results/train_seed444_20260410_055647.log`

| Metric | Value |
|--------|-------|
| model_params | 29,415,508 |
| raw_bpb | 1.1532 |
| int6_sw_bpb | 1.13678077 |
| step_avg_ms | 150.12 |
| steps | 3,997 |
| bytes_total | 16,065,029 (OVER 16MB cap by 65,029 bytes) |
| bytes_code | 122,265 |
| artifact_legal | NO |

Gate: beat 1.13867894 int6_sw_bpb — YES (-0.00190)
Gate: artifact <= 16,000,000 bytes — NO (missed by 65,029 bytes)

Structured metrics snapshot:
`crawler/2026-04-09_Trapper_Keeper_1/results/metrics_seed444_20260410_055647_recovered.tsv`

## Recovered disconnected run (8xH100, 600s, seed=444, 7F+3C, loop-aware GPTQ)

Status: [x] QUALITY PASS — NEAR-MISS SIZE FAIL

Command:
`RUNTIME_PYMINIFY=1 PYMINIFY_MODE=aggressive LOOP_AWARE_GPTQ=1 SKIP_GPTQ=0 SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-09_Trapper_Keeper_1/run_7f3c_brotli_gptq_pyminify.sh`

Recovered log:
`crawler/2026-04-09_Trapper_Keeper_1/results/train_seed444_20260410_065634_recovered_from_chat.log`

| Metric | Value |
|--------|-------|
| model_params | 29,415,508 |
| raw_bpb | 1.1530 |
| int6_sw_bpb | 1.13417607 |
| step_avg_ms | 150.08 |
| steps | 3,998 |
| bytes_total | 16,032,878 (OVER 16MB cap by 32,878 bytes) |
| bytes_code | 65,410 |
| artifact_legal | NO |

Gate: beat 1.13867894 int6_sw_bpb — YES (-0.00450)
Gate: artifact <= 16,000,000 bytes — NO (missed by 32,878 bytes)

Structured metrics snapshot:
`crawler/2026-04-09_Trapper_Keeper_1/results/metrics_seed444_20260410_065634_recovered_from_chat.tsv`

## Safe vs Aggressive gate (4xGPU, 300s wallclock)

| Arm | Config | steps_in_300s | step_ms | val_bpb | Verdict |
|-----|--------|---------------|---------|---------|---------|
| A0 | 3 loops, ROPE=9,1,1 | 548 | 547ms | 3.2425 | WINNER |
| A1 | 4 loops, ROPE=9,3,1,1 | 439 | 685ms | 3.3202 | LOSES both axes |

3 loops confirmed optimal for 8F+3C. 4th loop is pure waste under wallclock pressure.

## Screen evidence

| Source | Config | int6_sw | Environment |
|--------|--------|---------|-------------|
| Grid 2xGPU | 8F+3C | 1.39529 | 2xGPU, 1000 steps |
| Isolated 4xGPU | 8F+3C | 1.34632 | 4xGPU, 1000 steps |
| Grid control | 9F+1C | 1.41256 | 2xGPU, 1000 steps |
| **Production 8xH100** | **8F+3C** | **1.13527** | **8xH100, 600s, 3811 steps (no FA3)** |
