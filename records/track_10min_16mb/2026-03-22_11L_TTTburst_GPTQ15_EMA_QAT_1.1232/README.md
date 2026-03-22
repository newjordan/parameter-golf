## Record: 11L TTT Burst + GPTQ-15 + EMA + QAT

**val_bpb: 1.1232** (sliding window stride=64) | **15.68 MB** | 8xH100 SXM, 600s

### Key Innovations Over PR #414

| Change | PR #414 | This | Impact |
|--------|---------|------|--------|
| **TTT Burst** | None | 2-epoch replay of last 100 batches at 10% LR, QAT forced | Sharpens model on recent training data before EMA finalization |
| **GPTQ-15** | 5 clip percentiles | 15 clip percentiles per row, pick min MSE | Tighter quantization clips, less reconstruction error |

Everything else inherited from PR #414: EMA(0.997), warmdown 3500, Late QAT@0.15, int6+zstd-22.

### TTT Burst: Late-Stage QAT-Aware Sharpening

After the main training loop and before EMA application, we replay the last 100 training batches for 2 epochs at 10% of base LR with QAT forced on. EMA is updated during the burst so it absorbs the sharpened signal. This reduces the quantization tax by training the model to be robust to int6 noise right before finalization.

### GPTQ-15: Finer Clip Percentile Grid

Instead of 5 percentiles for int6 clip selection, we search 15 (from 0.998 to 1.0). More candidates = better MSE-optimal clip per weight row. Zero training cost, ~30s extra at export time.

### Results

| Seed | Steps | val_loss | Sliding BPB (s64) | Artifact |
|------|-------|----------|-------------------|----------|
| **1337** | 6991 | 1.9246 | **1.1232** | 15.68 MB |
| 42 | pending | pending | pending | pending |
| 2024 | pending | pending | pending | pending |

### Architecture

11L, 512d, 8H/4KV, MLP 3x (relu^2), U-Net skips, XSA4, Partial RoPE 16/64, LN Scale, VE128, SmearGate, BigramHash(2048), FA3, Muon WD=0.04, EMA(0.997), Tight SWA, Late QAT@0.15, TTT Burst(2ep/10%LR/QAT), int6+zstd-22, GPTQ-15.

### Run Command

```bash
SEED=1337 torchrun --nproc_per_node=8 train_gpt.py
```

### Test plan

- [x] Seed 1337 under 16MB (15.68 MB)
- [x] Seed 1337 trains in 600s on 8xH100
- [x] Post-quant roundtrip verified
- [ ] Seed 42 under 16MB
- [ ] Seed 2024 under 16MB
- [ ] 3-seed mean computed
- [x] train_gpt.py under 1500 lines
- [x] No TTT on validation data
