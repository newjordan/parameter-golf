# Hypothesis: Nightcrawler Cubed Folding

Date: 2026-04-10
Track: crawler
Parent: `nightcrawler_cubed` locked winner

## What changes

One variable vs the frozen `nightcrawler_cubed` winner:
- `LINKED_FLAT_RECUR_LAYERS=1`

Implementation detail:
- reuse the deepest encoder flat block inside each crawler loop
- no new block types
- same `3C` crawler body
- same loop index, same loop RoPE battery, same anchor/tap injection path

Everything else stays on the current winner stack.

## Why

This is the smallest test of “linked flat recurrence + crawler recurrence”.

Instead of adding a separate recurrence system, one existing flat block is
re-used inside each crawler pass. That lets the loop battery act on both:

- a flat recurrent refinement slice
- the existing crawler refinement body

The idea is to check whether the flat slice and crawler specialize into a
useful two-stage loop, without paying for a large architectural fork.

## Risks

- extra loop compute may cost enough steps to wipe out any quality gain
- reusing a flat block inside the crawler loop may want different GPTQ Hessians
- the flat recurrent slice may duplicate work the crawler already does

## First readout

Gate on seed `444` first. Watch all four:

- `int6_sw_bpb`
- artifact bytes
- `step_avg`
- GPTQ / prune behavior
