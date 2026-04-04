# Fractal Transformer + Gravity + AttnRes

## Summary

Replace the baseline's 9 unique transformer layers with **3 shared layers repeated across 3 loops**, yielding wider layers (864d vs 512d) within the same parameter budget. Add two mechanisms for depth coherence:

1. **Gravity** — learned auxiliary cross-entropy loss at each loop boundary, giving every loop direct supervision instead of relying on diluted backprop from the final output
2. **Attention Residuals (AttnRes)** — replace fixed U-Net skip connections with learned, input-dependent attention over previous loop outputs (from [Moonshot's AttnRes paper](https://arxiv.org/abs/2603.15031))

## Architecture

```
INPUT (1024 vocab)
  → Embedding (1024 × 864)
  → RMSNorm
  
LOOP 1 (broad):
  Layer A → Layer B → Layer C  (shared weights, loop_pos=0)
  GRAVITY: aux loss₁ (learned weight)
  Store output for AttnRes

LOOP 2 (refine):
  AttnRes: attend over [embedding, loop1_out]
  Layer A → Layer B → Layer C  (same weights, loop_pos=1)
  GRAVITY: aux loss₂ (learned weight)
  Store output for AttnRes

LOOP 3 (precision):
  AttnRes: attend over [embedding, loop1_out, loop2_out]
  Layer A → Layer B → Layer C  (same weights, loop_pos=2)
  FINAL LOSS (learned weight)

  → RMSNorm → lm_head (tied) → logit softcap → BPB
```

## Key Changes from Baseline

| Component | Baseline | This Submission |
|-----------|----------|----------------|
| Depth structure | 9 unique layers, U-Net skip | 3 shared × 3 loops |
| Model width | 512 | 864 |
| Skip connections | Fixed U-Net + resid_mix | AttnRes (learned, input-dependent) |
| Training signal | Single final loss | Gravity: learned aux loss per loop |
| Unique params | ~17M | ~16.6M (fewer, but wider) |

## What's Removed
- U-Net encoder/decoder split
- `skip_weights` parameter
- `resid_mix` parameter

## What's Added
- Loop position embeddings: 3 × 864 = 2,592 params
- AttnRes query vectors: 9 × 864 = 7,776 params  
- Gravity weight scalars: 3 params
- **Total added: ~10,371 params**
- **Net savings: ~3,453 params** (reinvested into wider dim)

## Local Results (DGX Spark GB10, 300 steps, 1 shard)

| Config | val_bpb | Δ vs baseline |
|--------|--------:|----------:|
| Baseline (9 unique, 512d) | 2.7927 | — |
| Fractal only (3×3, 864d) | 2.5953 | -0.1975 |
| Fractal + Gravity | 2.6149 | -0.1779 |
| Fractal + Gravity + AttnRes | 2.6084 | -0.1843 |

**Note:** Local runs use AdamW (not Muon), no torch.compile, 300 steps on 1 shard.
These are directional — absolute BPB values are not comparable to competition scale.
Gravity is expected to improve with more training steps (see RESULTS.md).

## Theoretical Basis

- **Weight sharing / depth recurrence**: explicitly encouraged by the challenge README as a promising direction for parameter-constrained settings
- **AttnRes**: Moonshot paper (arxiv:2603.15031) shows 1.25× compute equivalent at near-zero parameter cost by replacing fixed residual connections with learned attention over depth
- **Deep supervision / auxiliary losses**: well-established technique for improving gradient flow in deep networks; especially beneficial for weight-shared architectures where the same weights receive gradient from multiple depths

## Status

**Pending cloud validation.** Needs 8×H100 10-minute run with Muon optimizer and torch.compile to produce official BPB numbers.

## References

- [Attention Residuals (Moonshot, 2026)](https://arxiv.org/abs/2603.15031)
- [Universal Transformers (Dehghani et al., 2019)](https://arxiv.org/abs/1807.03819)
- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
