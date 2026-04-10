# Hypothesis: Trapper_Keeper_1

Date: 2026-04-09
Track: crawler
Parent: records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/

## What changes

TWO env vars vs BWX 9F production:
- NUM_FLAT_LAYERS: 9 → 8
- NUM_CRAWLER_LAYERS: 1 → 3

Everything else identical. Same train_gpt.py (zero diff).

## Why

Layer relationship grid (20-arm, 2xGPU, 1000 steps) mapped the full
flat(5-9) × crawler(1-4) surface. Findings:

- 8F+3C is the quality peak: 1.39529 int6_sw (vs 1.41256 for 9F+1C control)
- 3 crawler layers matches the 3 loop count — C=loops symmetry
- 4C reverses at 3 loops (9F+4C = 1.40689, worse than 3C)
- Below 7F quality drops off — crawler can't compensate for too few flat layers
- 8F+3C isolated on 4xGPU confirmed: 1.34632 int6_sw (beats 9F+1C by -0.0085)

The 9th flat layer does nothing when 3 crawlers are present.
Size: 15.00MB (legal, 1MB headroom).

## Gate target

Beat 1.13867894 int6_sw_bpb (seed 444) with artifact <= 16,000,000 bytes.
Confirm on seed 300.
