# Ablation: BW_9F2C
Date: 2026-04-09
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## Gate (1-GPU, 2000 steps, seed=444)
Status: [x] skipped — promoted from corpus ablation A07 screen (delta −0.0119)
Notes: Gate equivalent was 16-arm corpus ablation v1, arm A07 (NUM_CRAWL=2). Strongest signal in screen.

## Full run (8×H100, 600s, seed=444)
Status: [x] pass (quality) / [x] fail (size)
step_avg: 137.76ms
steps: 4,356
val_bpb (live weights, SKIP_EMA=1): 1.1484
int6_roundtrip_exact: 1.15604305
int6_sw_bpb: 1.13189759
artifact_bytes (int6+brotli): 16,857,961 (OVER 16MB cap)
Code size: 122,265
model_params: 30,204,508
peak_memory: 33,300 MiB
SKIP_GPTQ=1, SKIP_EMA=1
Notes: Quality beats BWX 9F leader (1.13867894) by −0.00678 on int6_sw. But artifact busts 16MB. step_avg 137.76ms is ~2× the 74.68ms target due to extra crawler layer.

## Confirmation (8×H100, 600s, seed=300)
Status: [ ] pending — blocked by size cap failure
int6_sw_bpb:
artifact_bytes:
