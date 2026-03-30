# Rascal: Rat Rod v1 + Coprime Loader

**Track:** 10-minute / 16MB
**Hardware:** 8×H100 SXM
**Date:** 2026-03-30

## Summary

Quality base neural model submission built on the Rat Rod v1 stack with a coprime-stride multi-shard block loader for improved training data diversity.

No trigram oracle, no n-gram eval, no MTP — pure base model quality.

## Architecture

| Component | Setting |
|---|---|
| Layers | 11 (flat) |
| Attention | GQA (8 heads / 4 KV heads) |
| XSA | All 11 layers (XSA_LAST_N=11) |
| Optimizer | Parallel Muon |
| Bigram vocab | 2048 |
| RoPE dims | 16 |
| MLP kernel | eager (LeakyReLU²) |
| SWA | every 50 steps |
| Late QAT | ~step 6080, scale ~0.15 |
| Compile | enabled, mode=default, fullgraph=1 |
| Embeddings | tied, embed_lr=0.035 |
| Model params | 26,993,756 |

## Loader

Coprime-stride block loader (`LOADER_MODE=coprime`). Each shard slot advances through training shards using a stride computed as the nearest integer coprime to the shard count, ensuring full coverage with minimal repetition per batch window.

| Param | Value |
|---|---|
| Shards | 80 |
| Shards per batch | 1 |
| Shard hold steps | 64 |

## Results

3-seed run, all eager MLP kernel, compile=default fullgraph=1.

| Seed | Sliding Exact BPB | Post-EMA BPB | Steps | Step avg |
|---|---|---|---|---|
| 300 | 1.10995812 | 1.1334 | 6599 | 90.94ms |
| 42 | 1.10960162 | 1.1332 | 6594 | 91.01ms |
| 444 | 1.10910521 | 1.1326 | 6590 | 91.06ms |
| **Mean** | **1.10955498** | **1.1331** | | |
| **Std** | **0.000428** | | | |

## Reproducing

```bash
# single seed
SEED=42 bash experiments/JunkRat_X/run.sh

# override wallclock cap
MAX_WALLCLOCK_SECONDS=600 SEED=300 bash experiments/JunkRat_X/run.sh
```

## Training Environment

- PyTorch + FlashAttention 3 (Hopper)
- 8×H100 SXM, torchrun standalone
- Batch tokens: 786,432 (seq_len=2048 × 384)
- Peak memory: ~22.9GB allocated per GPU

## Artifact Size

- Code: ~103KB (`train_gpt.py`)
- Model: TBD (late QAT int6 + zstd compression)
- Total: TBD / 16,000,000 bytes
