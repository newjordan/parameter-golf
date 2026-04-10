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

## Live legal run (8xH100, 600s, seed=444, 7F+3C, GPTQ + pyminify + selective prune)

Status: [x] QUALITY PASS — LEGAL ARTIFACT — confirmed on seeds 300 and 4

Command:
`SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-09_Trapper_Keeper_1/run_7f3c_brotli_gptq_pyminify_prune.sh`

Raw live log:
`crawler/2026-04-09_Trapper_Keeper_1/results/train_seed444_20260410_205837.log`

Safe frozen copy:
`crawler/2026-04-09_Trapper_Keeper_1/results/best_crawler_SOTA_CRAWLER_SAFE_seed444_20260410_205837.log`

| Metric | Value |
|--------|-------|
| model_params | 29,415,508 |
| raw_bpb | 1.1532 |
| int6_sw_bpb | 1.13541288 |
| step_avg_ms | 149.15 |
| steps | 4,023 |
| bytes_total | 15,902,698 (LEGAL, 97,302 bytes under cap) |
| bytes_code | 67,089 |
| artifact_legal | YES |

Gate: beat 1.13867894 int6_sw_bpb — YES (-0.00327)
Gate: artifact <= 16,000,000 bytes — YES (97,302 bytes headroom)

Selective prune note:
`selective_prune_int6 enabled target:16000000 pre_total:16016398 post_total:15902698 excess_pre:16398 values_pruned:393328`

Tradeoff vs unpruned near-miss:
- Saved 130,180 bytes (`16,032,878` → `15,902,698`)
- Gave back 0.00123681 int6_sw_bpb (`1.13417607` → `1.13541288`)
- Still beats BWX 9F by 0.00326606 on seed 444 while staying legal

Structured metrics snapshot:
`crawler/2026-04-09_Trapper_Keeper_1/results/metrics_seed444_20260410_205837.tsv`

## Confirmation run (8xH100, 600s, seed=300, 7F+3C, GPTQ + pyminify + selective prune)

Status: [x] CONFIRMS — LEGAL ARTIFACT

Raw live log:
`crawler/2026-04-09_Trapper_Keeper_1/results/train_seed300_20260410_211710.log`

Safe frozen copy:
`crawler/2026-04-09_Trapper_Keeper_1/results/best_crawler_SOTA_CRAWLER_SAFE_seed300_20260410_211710.log`

| Metric | Value |
|--------|-------|
| model_params | 29,415,508 |
| raw_bpb | 1.1555 |
| int6_sw_bpb | 1.13853446 |
| step_avg_ms | 149.78 |
| steps | 4,006 |
| bytes_total | 15,851,974 (LEGAL, 148,026 bytes under cap) |
| bytes_code | 67,089 |
| artifact_legal | YES |

Seed-300 delta vs BWX 9F:
- `1.13867894` → `1.13853446` (`-0.00014448`)

Structured metrics snapshot:
`crawler/2026-04-09_Trapper_Keeper_1/results/metrics_seed300_20260410_211710.tsv`

## Third required run (8xH100, 600s, seed=4, 7F+3C, GPTQ + pyminify + selective prune)

Status: [x] PASS — LEGAL ARTIFACT

Raw live log:
`crawler/2026-04-09_Trapper_Keeper_1/results/train_seed4_20260410_213324.log`

Safe frozen copy:
`crawler/2026-04-09_Trapper_Keeper_1/results/best_crawler_SOTA_CRAWLER_SAFE_seed4_20260410_213324.log`

| Metric | Value |
|--------|-------|
| model_params | 29,415,508 |
| raw_bpb | 1.1528 |
| int6_sw_bpb | 1.13536063 |
| step_avg_ms | 149.57 |
| steps | 4,012 |
| bytes_total | 15,844,157 (LEGAL, 155,843 bytes under cap) |
| bytes_code | 67,089 |
| artifact_legal | YES |

Structured metrics snapshot:
`crawler/2026-04-09_Trapper_Keeper_1/results/metrics_seed4_20260410_213324.tsv`

## Three-seed verdict

Status: [x] CURRENT IN-TREE WINNER

| Seed | int6_sw_bpb | bytes_total | legal |
|------|-------------|-------------|-------|
| 444 | 1.13541288 | 15,902,698 | YES |
| 300 | 1.13853446 | 15,851,974 | YES |
| 4 | 1.13536063 | 15,844,157 | YES |
| mean | 1.13643599 | 15,902,698 (max) | YES |

All three required seeds are legal and all three beat the BWX 9F in-tree leader.
This variant is now the locked-in crawler winner pending records-folder packaging.

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
