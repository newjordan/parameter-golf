# Pocket_Bandit — Compact Architecture Garage

Date: 2026-03-29

## Mission
Keep the Bandit line under constant pressure as the compact-architecture garage, not as a worse copy of the flat base lane.

This garage answers one question:
how do we spend Bandit's remaining budget margin better than everyone else?

## Anchor

Current public-local reference:
- Bandit (`ClownCar + X-WING ngram9`) at `0.4961 BPB`
- honest base sliding at `1.1867`
- mean size about `9.2 MB`

Reference file:
- `records/track_10min_16mb/2026-03-29_Bandit_ClownCar_X_CubricNgram9_8xH100/README.md`

## Why This Garage Exists

Bandit matters for a different reason than Rat Rod:
- Rat Rod is the better honest base-model anchor
- Bandit is the better compact-architecture proof
- the unused size budget is still strategic headroom

So Pocket_Bandit is where we test:
- width/depth spending inside the crawler family
- transfer of proven base-lane systems wins
- compact-architecture ideas that do not require a 15-16 MB stack

## Imported Hypotheses

### H1: take only proven systems wins from Junkyard_Rat

If the `#1060`-style loader or the `#1072` Triton path wins on the clean base lane, transfer it here.

Decision rule:
- no direct import into Pocket_Bandit until it survives on Junkyard_Rat first

### H2: spend the remaining memory on architecture, not ceremony

Primary levers:
- width
- depth split
- crawler loop count
- instruction bottleneck size

Decision rule:
- prefer any change that improves compact BPB per MB, not just raw base quality

## Immediate Queue

1. Keep `Bandit_Wagon` as the current execution sheet for headroom sweeps.
2. When Junkyard_Rat produces a winning loader or Triton result, port that change into the Bandit stack.
3. Re-run width and depth arms after the systems import, not before.

## Relationship To Existing Notes

`Bandit_Wagon` remains the tactical ablation sheet.

`Pocket_Bandit` is the higher-level garage identity:
- compact architecture
- size efficiency
- transfer target for clean base-lane wins
