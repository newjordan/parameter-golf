# Ablation: Crawler_Katta

Date: 2026-04-07  
Track: crawler  
Parent: legs/2026-04-07_BW22_LoopDepth_9F/

## Gate (8xGPU, 2000 steps, seed=444)

Status: [ ] pending  [ ] pass  [ ] fail

| Arm | Loops | Rope Scales | Solver | RK Heads | RK Battery | model_params | raw_bpb | int6_sw_bpb | step_ms | bytes | delta_vs_ctrl |
|-----|-------|-------------|--------|----------|------------|--------------|---------|-------------|---------|-------|---------------|
| A0_ctrl_euler_l3 | 3 | 9,1,1 | euler | 2 | 1,1,1,1 | | | | | | -- |
| A1_rk2_fast_l2 | 2 | 9,1 | rk2_fast | 2 | 1.0,1.2,1.0,1.0 | | | | | | |
| A2_rk4_fast_l2 | 2 | 9,1 | rk4_fast | 2 | 1.0,1.15,1.0,1.05 | | | | | | |
| A3_rk24_hybrid_l2 | 2 | 9,1 | rk24_hybrid | 2 | 1.0,1.1,1.0,1.0 | | | | | | |
| A4_rk24_hybrid_l3 | 3 | 9,3,1 | rk24_hybrid | 2 | 1.0,1.1,1.0,1.0 | | | | | | |

Notes:
