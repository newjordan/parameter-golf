# Results: BW22_LoopDepth_9F

Date: 2026-04-07
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/
Gate run: seed=444, 8xGPU, 2000 steps per arm

## Verdict

[x] PROMOTES  [ ] DOES NOT PROMOTE

## Gate Summary

| Arm | LOOPS | ROPE_SCALES | int6_sw_bpb | step_ms | delta_vs_ctrl |
|-----|-------|-------------|-------------|---------|---------------|
| A0_ctrl | 3 | 9,1,1 | 1.24352055 | 110.28 | -- |
| A1_loop4_naive | 4 | 9,1,1,1 | 1.24255272 | 119.82 | -0.00096783 |
| A2_loop4_battery | 4 | 9,3,1,1 | 1.24259530 | 119.58 | -0.00092525 |
| A3_loop5_battery | 5 | 9,3,1,1,1 | 1.24091238 | 128.88 | -0.00260817 |
| A4_loop5_prog | 5 | 9,5,3,1,1 | 1.24176280 | 128.99 | -0.00175775 |

## What we learned

- All ablation arms improved `int6_sw_bpb` vs control.
- Quality gain scaled with depth; best was `A3_loop5_battery`.
- Throughput cost was significant:
- loop4 arms: ~8.4-8.7% slower than control
- loop5 arms: ~16.9-17.0% slower than control
- Progressive rope battery at loop5 (`A4`) gave the smallest artifact size.

## 4-hour run recommendation

- Quality-first: `A3_loop5_battery`.
- Compression-first with strong quality: `A4_loop5_prog`.
- If throughput is the primary constraint, keep loop3 control and test RK2/RK4 hybrid in `Crawler_Katta`.
