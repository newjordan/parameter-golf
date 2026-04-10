# Results: BW_9F2C
Date: 2026-04-09
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## Verdict
[x] DOES NOT PROMOTE — quality pass but artifact over 16MB cap

## Scores
| Seed | int6_sw_bpb | artifact | vs leader |
|------|-------------|----------|-----------|
| 444  | 1.13189759  | 16,857,961 bytes | −0.00678 (beats) |
| 300  | —           | —        | blocked by size fail |
| mean | —           | —        | — |

Leader: BWX 9F = 1.13867894 int6_sw_bpb, 15,239,617 bytes

## What we learned
- NUM_CRAWLER_LAYERS=2 (9F+2C) delivers real quality: −0.00678 vs leader on int6_sw_bpb. The corpus ablation A07 signal (−0.0119 at 1500 steps) held direction at production scale.
- But the extra crawler layer adds ~1MB of params, pushing artifact to 16.86MB — 860KB over cap. GPTQ was skipped (SKIP_GPTQ=1), so GPTQ could potentially recover size, but adds wallclock cost.
- step_avg 137.76ms is nearly 2× the 74.68ms target. Only 4,356 steps in 600s vs ~8,000 at target pace. More steps would improve quality further but wallclock is the hard constraint.
- The 2nd crawler layer is expensive: +params, +step_time, +artifact size. Quality gain is real but the budget doesn't fit.

## Next hypothesis
- If size is the only blocker, test GPTQ on 9F+2C to see if quantization can bring artifact under 16MB. But step_avg penalty means fewer training steps, so net quality may not hold.
- Alternatively, 8F+3C (TK1) showed similar quality (1.13527) with different size/speed tradeoffs — brotli recompress on TK1 may be the faster path to a promotable result.
