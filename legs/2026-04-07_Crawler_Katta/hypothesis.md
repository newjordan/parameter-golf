# Hypothesis: Crawler_Katta

Date: 2026-04-07  
Track: crawler  
Parent: legs/2026-04-07_BW22_LoopDepth_9F/

## What changes

Add a fast Runge-Kutta crawler solver to the loop core:

- `CRAWLER_SOLVER=euler` (baseline)
- `CRAWLER_SOLVER=rk2_fast`
- `CRAWLER_SOLVER=rk4_fast`
- `CRAWLER_SOLVER=rk24_hybrid` (new RK2/RK4 hybrid)

The RK variants keep one full crawler pass per loop and approximate intermediate RK
stages with lightweight per-head mixers (`CRAWLER_RK_FAST_HEADS`) plus recurrent
delta carry (`CRAWLER_RK_RECUR_GAIN_INIT`) and stage battery gains
(`CRAWLER_RK_BATTERY`).

## Why

Goal is better quality-per-step under a fixed wallclock budget. If RK variants with
fewer loops match or beat control BPB while reducing `step_ms`, they are better
4-hour production candidates.

## Gate target

- Primary: lower `step_ms` vs control.
- Secondary: non-regressive `int6_sw_bpb` vs control.
- Any arm with better `int6_sw_bpb` and lower `step_ms` is a promote candidate.
