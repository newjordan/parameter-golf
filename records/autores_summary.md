# Vortex-Helix fused kernel — autoresearch summary

## Kernel config sweep (vortex_fused)

Configs evaluated: 144
- **Best fwd_ms** = 0.461 ms @ block=64, fwd_warps=8, fwd_stages=3
- **Best bwd_ms** = 4.868 ms @ block=128, bwd_warps=8, bwd_stages=2

Top 8 by fwd_ms:

| block | fw_w | fw_s | bw_w | bw_s | fwd_ms | bwd_ms | mem MB |
|---|---|---|---|---|---|---|---|
| 64 | 8 | 3 | 4 | 1 | 0.461 | 5.262 | 208.4 |
| 64 | 8 | 3 | 8 | 1 | 0.461 | 8.361 | 208.4 |
| 64 | 8 | 3 | 8 | 2 | 0.461 | 8.340 | 208.4 |
| 64 | 8 | 3 | 4 | 2 | 0.462 | 5.211 | 208.4 |
| 64 | 8 | 2 | 8 | 2 | 0.465 | 8.348 | 208.4 |
| 64 | 8 | 2 | 8 | 1 | 0.466 | 8.359 | 208.4 |
| 64 | 8 | 2 | 4 | 1 | 0.467 | 5.307 | 208.4 |
| 64 | 8 | 2 | 4 | 2 | 0.467 | 5.223 | 208.4 |

Top 8 by bwd_ms:

| block | fw_w | fw_s | bw_w | bw_s | fwd_ms | bwd_ms | mem MB |
|---|---|---|---|---|---|---|---|
| 128 | 8 | 2 | 8 | 2 | 0.489 | 4.868 | 208.4 |
| 32 | 4 | 2 | 8 | 2 | 0.501 | 4.874 | 208.6 |
| 128 | 8 | 2 | 8 | 2 | 0.498 | 4.880 | 208.4 |
| 128 | 8 | 3 | 8 | 2 | 0.486 | 4.898 | 208.4 |
| 32 | 4 | 3 | 8 | 2 | 0.510 | 4.900 | 208.6 |
| 32 | 4 | 1 | 8 | 2 | 0.560 | 4.906 | 208.6 |
| 128 | 8 | 3 | 8 | 2 | 0.489 | 4.925 | 208.4 |
| 128 | 8 | 1 | 8 | 2 | 0.530 | 4.928 | 208.4 |

## Seqlen scaling (fwd_ms / bwd_ms, all impls)

### fwd_ms
| T | eager_vortex | sdpa_cudnn_attn_only | sdpa_cudnn_vortex | sdpa_flash_attn_only | sdpa_flash_vortex | sdpa_memeff_attn_only | vortex_fused |
|---|---|---|---|---|---|---|---|
| 512 | 0.721 | 0.049 | 0.660 | 0.046 | 0.516 | 0.049 | 0.146 |
| 1024 | 1.662 | 0.060 | 0.933 | 0.079 | 0.889 | 0.108 | 0.278 |
| 2048 | 4.995 | 0.114 | 1.838 | 0.189 | 1.893 | 0.282 | 0.538 |
| 4096 | 16.715 | 0.288 | 3.527 | 0.526 | 3.750 | 0.915 | 1.193 |
| 8192 | 80.448 | 0.945 | 10.670 | 1.730 | 8.557 | 3.713 | 3.042 |

### bwd_ms
| T | eager_vortex | sdpa_cudnn_attn_only | sdpa_cudnn_vortex | sdpa_flash_attn_only | sdpa_flash_vortex | sdpa_memeff_attn_only | vortex_fused |
|---|---|---|---|---|---|---|---|
| 512 | 4.465 | 0.228 | 3.264 | 0.189 | 2.478 | 0.240 | 1.229 |
| 1024 | 5.007 | 0.283 | 3.367 | 0.296 | 3.167 | 0.509 | 2.579 |
| 2048 | 13.355 | 0.418 | 5.166 | 0.719 | 5.484 | 1.328 | 5.329 |
| 4096 | 49.358 | 1.103 | 9.984 | 2.011 | 10.917 | 4.117 | 12.695 |
| 8192 | 229.494 | 3.739 | 28.847 | 6.662 | 35.382 | 19.070 | 34.221 |

## Chaos-stream ablation (chaos_iters vs impls)

_Note: vortex_fused is hard-coded at chaos_iters=5; only eager/sdpa-vortex rows vary._

### fwd_ms
| chaos_iters | eager_vortex | sdpa_cudnn_attn_only | sdpa_cudnn_vortex | sdpa_flash_attn_only | sdpa_flash_vortex | sdpa_memeff_attn_only | vortex_fused |
|---|---|---|---|---|---|---|---|
| 0 | 3.803 | 0.118 | 0.605 | 0.194 | 0.677 | 0.290 | 0.544 |
| 1 | 4.043 | 0.115 | 0.862 | 0.191 | 0.923 | 0.288 | 0.542 |
| 3 | 4.510 | 0.114 | 1.341 | 0.195 | 1.425 | 0.293 | 0.539 |
| 5 | 4.991 | 0.115 | 1.881 | 0.193 | 1.884 | 0.290 | 0.539 |
| 7 | 5.468 | 0.114 | 2.345 | 0.192 | 2.366 | 0.288 | 0.540 |

### bwd_ms
| chaos_iters | eager_vortex | sdpa_cudnn_attn_only | sdpa_cudnn_vortex | sdpa_flash_attn_only | sdpa_flash_vortex | sdpa_memeff_attn_only | vortex_fused |
|---|---|---|---|---|---|---|---|
| 0 | 9.923 | 0.427 | 1.645 | 0.707 | 1.938 | 1.304 | 5.322 |
| 1 | 10.618 | 0.431 | 2.404 | 0.700 | 2.692 | 1.302 | 5.325 |
| 3 | 11.967 | 0.416 | 3.780 | 0.710 | 4.067 | 1.301 | 5.315 |
| 5 | 13.337 | 0.450 | 5.215 | 0.704 | 5.420 | 1.310 | 5.304 |
| 7 | 14.716 | 0.436 | 6.533 | 0.699 | 6.978 | 1.309 | 5.269 |
