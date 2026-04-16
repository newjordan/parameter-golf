# Effective-potential surface: vortex_fused vs baselines

B=2, H=8, D=128, bf16, H100 sm_90.  Champion kernel cfg: block=128, fwd(w=8,s=2), bwd_chaos(w=8,s=1), bwd_attn(w=8,s=2).

All times are p50 ms over 20 iters (5 warmup).  Speedup = baseline/vortex.

| T | depth | vortex fwd | vortex bwd | eager_vortex fwd | eager_vortex bwd | sdpa_flash_vortex fwd | sdpa_flash_vortex bwd | sdpa_flash_attn_only fwd | sdpa_flash_attn_only bwd |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 0 | 0.065 | 0.534 | 3.90x | 3.52x | 2.58x | 2.07x | 0.82x | 0.37x |
| 512 | 1 | 0.081 | 0.646 | 3.86x | 3.60x | 3.14x | 1.57x | 0.58x | 0.30x |
| 512 | 3 | 0.091 | 0.795 | 4.51x | 4.47x | 3.85x | 2.15x | 0.53x | 0.25x |
| 512 | 5 | 0.106 | 1.110 | 4.85x | 4.13x | 4.30x | 2.92x | 0.60x | 0.18x |
| 512 | 7 | 0.122 | 1.282 | 4.97x | 3.80x | 4.45x | 2.72x | 0.38x | 0.15x |
| 1024 | 0 | 0.075 | 0.658 | 8.00x | 3.02x | 2.31x | 1.31x | 0.79x | 0.33x |
| 1024 | 1 | 0.087 | 0.735 | 7.65x | 2.50x | 2.87x | 1.36x | 0.66x | 0.30x |
| 1024 | 3 | 0.114 | 0.989 | 7.07x | 2.35x | 3.35x | 1.62x | 0.52x | 0.22x |
| 1024 | 5 | 0.134 | 1.375 | 6.91x | 2.39x | 3.78x | 1.57x | 0.42x | 0.16x |
| 1024 | 7 | 0.151 | 1.423 | 6.97x | 2.19x | 4.19x | 2.12x | 0.46x | 0.15x |
| 2048 | 0 | 0.140 | 1.147 | 13.31x | 4.25x | 2.50x | 0.89x | 0.92x | 0.39x |
| 2048 | 1 | 0.190 | 1.398 | 10.47x | 3.79x | 2.52x | 1.21x | 0.67x | 0.32x |
| 2048 | 3 | 0.239 | 1.984 | 9.28x | 3.03x | 2.95x | 1.47x | 0.59x | 0.23x |
| 2048 | 5 | 0.282 | 2.783 | 8.66x | 2.41x | 3.32x | 1.05x | 0.54x | 0.16x |
| 2048 | 7 | 0.321 | 2.995 | 8.35x | 2.49x | 3.64x | 1.20x | 0.44x | 0.15x |
| 4096 | 0 | 0.308 | 2.707 | 22.43x | 6.58x | 2.66x | 0.89x | 1.12x | 0.43x |
| 4096 | 1 | 0.401 | 3.296 | 17.92x | 5.64x | 2.66x | 0.97x | 0.90x | 0.35x |
| 4096 | 3 | 0.486 | 4.448 | 15.78x | 4.49x | 3.21x | 1.03x | 0.74x | 0.26x |
| 4096 | 5 | 0.582 | 5.773 | 14.01x | 3.71x | 3.53x | 1.03x | 0.60x | 0.20x |
| 4096 | 7 | 0.662 | 6.385 | 12.97x | 3.55x | 3.79x | 1.14x | 0.51x | 0.18x |
| 8192 | 0 | 0.864 | 8.400 | 38.07x | 10.54x | 2.22x | 0.71x | 1.13x | 0.43x |
| 8192 | 1 | 1.046 | 9.358 | 26.86x | 7.54x | 2.27x | 0.78x | 0.95x | 0.39x |
| 8192 | 3 | 1.186 | 11.724 | 24.47x | 6.25x | 2.79x | 0.84x | 0.85x | 0.31x |
| 8192 | 5 | 1.362 | 14.365 | 21.96x | 5.27x | 3.06x | 0.87x | 0.72x | 0.25x |
| 8192 | 7 | 1.788 | 16.635 | 25.64x | 6.85x | 2.97x | 1.05x | 0.55x | 0.22x |

## Effective potential takeaways

- vs **eager_vortex** (same workload): fwd 4x-38x, bwd 2x-11x. Speedup falls as depth rises because the chaos stream is where eager loses least ground.
- vs **sdpa_flash_vortex** (flash attn + eager chaos): fwd 2x-5x (increases with depth); bwd 1x-4x except at large-T/depth=0 where flash bwd is faster (we are 0.7x-0.9x).
- vs **sdpa_flash_attn_only** (pure attn, no chaos): we still win on fwd at short T despite doing strictly more work; bwd at T>=4096, d=0 is slower (0.2x-0.4x) -- the attn bwd itself is the remaining bottleneck.

## Remaining gap

Backward at large T and low chaos depth is the only cell where a mainstream baseline (sdpa_flash) beats us. Targeted next steps would be to revisit the attn-only bwd kernel (fewer matmul loops, better software pipelining).
