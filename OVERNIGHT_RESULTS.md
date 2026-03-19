# Parameter Golf — Overnight Sweep Results
**DGX Spark GB10**

## Phase 1: Dimension Sweep (baseline mode, 300 steps, 1M eval tokens)

| Exp | layers | dim | heads | kv | mlp | val_bpb | train_loss@300 | ms/step |
|-----|--------|-----|-------|----|----|---------|----------------|---------|
| D1_baseline | 9 | 512 | 8 | 4 | 2 | **2.775959** | 4.6304 | 167.0ms |
| D2_dim576 | 9 | 576 | 8 | 4 | 2 | **2.733266** | 4.5644 | 208.1ms |
| D3_dim448 | 9 | 448 | 8 | 4 | 2 | **2.845552** | 4.7388 | 151.4ms |
| D4_dim640 | 9 | 640 | 8 | 4 | 2 | **2.692171** | 4.5040 | 232.1ms |
| D5_mlp3 | 9 | 512 | 8 | 4 | 3 | **2.769278** | 4.6280 | 197.5ms |
| D6_kv8 | 9 | 512 | 8 | 8 | 2 | **2.772609** | 4.6316 | 183.5ms |
| D7_kv2 | 9 | 512 | 8 | 2 | 2 | **2.787523** | 4.6476 | 160.2ms |
| D8_dim512_h4 | 9 | 512 | 4 | 2 | 2 | **2.761735** | 4.6064 | 165.4ms |

**Phase 1 Best: D4_dim640 (9L 640d 8H kv4 mlp2 ) — val_bpb=2.692171**


## Phase 2: LR & Schedule Sweep (fixed 9L 512d baseline dims)

| Exp | warmup | lr | val_bpb | train_loss@300 | ms/step |
|-----|--------|-----|---------|----------------|---------|
| L1_baseline | 20 | 3e-4 | **2.777890** | 4.6329 | 167.8ms |
| L2_lr5e4 | 20 | 5e-4 | **2.652223** | 4.4260 | 167.8ms |
| L3_lr2e4 | 20 | 2e-4 | **2.911767** | 4.8407 | 167.6ms |
| L4_lr1e3 | 20 | 1e-3 | **2.550388** | 4.2657 | 168.1ms |
| L5_warmup5 | 5 | 3e-4 | **2.759689** | 4.6014 | 168.1ms |
| L6_warmup50 | 50 | 3e-4 | **2.768084** | 4.6195 | 167.5ms |
| L7_warmup10 | 10 | 3e-4 | **2.767279** | 4.6202 | 167.9ms |
| L8_lr5e4_w10 | 10 | 5e-4 | **2.669585** | 4.4644 | 168.0ms |
| L9_lr1e3_w30 | 30 | 1e-3 | **2.479826** | 4.1610 | 168.3ms |

**Phase 2 Best so far: L9_lr1e3_w30 (warmup=30 lr=1e-3) — val_bpb=2.479826**


## Phase 3: Batch Size Sweep

| Exp | batch_tokens | val_bpb | train_loss@300 | ms/step |
|-----|-------------|---------|----------------|---------|
| B1_baseline | 16384 | **2.786178** | 4.6475 | 168.3ms |
| B2_batch32k | 32768 | **2.720247** | 4.5201 | 346.6ms |
| B3_batch8k | 8192 | **2.839658** | 4.8468 | 77.4ms |
| B4_batch64k | 65536 | **2.678867** |  | 686.4ms |


## FINAL SUMMARY

**Global Best: L9_lr1e3_w30 (warmup=30 lr=1e-3)**
**val_bpb: 2.479826**

Baseline target to beat on H100: **1.2244** (need ≤1.2194)
