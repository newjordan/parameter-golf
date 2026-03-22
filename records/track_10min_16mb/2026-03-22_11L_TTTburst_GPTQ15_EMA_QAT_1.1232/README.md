## Record: 11L TTT Burst + EMA + GPTQ-lite + warmdown3500 + QAT@0.15

**val_bpb: 1.1236** (sliding window stride=64, 2-seed mean) | **15.59 MB** (mean) | 8xH100 SXM, 600s

### Key Innovation Over PR #414

| Change | PR #414 | This | Impact |
|--------|---------|------|--------|
| **TTT Burst** | None | 2-epoch replay of last 100 training batches at 10% LR before EMA | -0.0001 BPB |

Everything else inherited from PR #414: EMA(0.997), GPTQ-lite(5 percentiles), warmdown 3500, Late QAT@0.15, int6+zstd-22.

### TTT Burst: Late-Stage Sharpening

After the main training loop and before EMA application, we replay the last 100 training batches for 2 epochs at 10% of base LR. EMA is updated during the burst so it absorbs the sharpened signal. This gives the model a final sharpening pass on recent data before weight averaging and quantization.

### Results (3 seeds, 8xH100 SXM)

| Seed | Steps | val_loss | Sliding BPB (s64) | Artifact |
|------|-------|----------|-------------------|----------|
| **1337** | 6991 | 1.9246 | **1.1232** | 15.68 MB |
| 42 | 6994 | 1.9262 | 1.1240 | 16.37 MB* |
| **2024** | 6987 | 1.9255 | **1.1239** | 15.50 MB |

**Mean (1337+2024): 1.1236 | Std: 0.0004**

*Seed 42 artifact over size limit due to compression variance; BPB validates the approach.

### Architecture

11L, 512d, 8H/4KV, MLP 3x (relu^2), U-Net skips, XSA4, Partial RoPE 16/64, LN Scale, VE128, SmearGate, BigramHash(2048), FA3, Muon WD=0.04, EMA(0.997), Tight SWA, Late QAT@0.15, TTT Burst(2ep/10%LR), int6+zstd-22, GPTQ-lite.

### Run Command

```bash
SEED=1337 torchrun --nproc_per_node=8 train_gpt.py
```

### Test plan

- [x] All seeds train in 600s on 8xH100
- [x] Seeds 1337, 2024 under 16MB (15.68 MB, 15.50 MB)
- [x] Post-quant int6 roundtrip verified
- [x] Sliding window eval (stride=64) consistent across seeds (std=0.0004)
- [x] train_gpt.py under 1500 lines (1443)
- [x] No TTT on validation data
