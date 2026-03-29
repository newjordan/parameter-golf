# Junkyard_Rat Triton Track

Date: 2026-03-29

## Mission

Turn `JR-02` from a one-off kernel experiment into a controlled Triton optimization track for the current `JR-01` winner.

This subfolder exists to separate:
- live base-lane winners
- kernel engineering work
- numerics compensation work after kernel changes

## Why This Track Exists

`JR-01` already proved the loader idea is real.

The Triton question is different:
- not "is Triton cool"
- but "can a custom kernel on our exact MLP path improve step time or quality on the winning stack"

This architecture is not exotic, but it is tightly tuned:
- banked FP32 weights
- BF16 runtime math
- `linear -> leaky_relu(0.5) -> square -> linear`
- layerwise residual scaling and mixing

That means a kernel can change:
- dataflow
- launch behavior
- math ordering
- effective numerics in the MLP branch

So Triton needs its own track.

## Current State

### Winner under test

`JR-01`:
- coprime loader
- ~`91.00ms` step time
- `1.11056240` sliding BPB

### Active Triton candidate

`JR-02`:
- `MLP_KERNEL_MODE=triton_act`
- custom Triton activation kernel in the real MLP branch
- same loader winner underneath

## Core Hypotheses

### H1: activation-kernel path is a live optimization surface

The first real Triton kernel path is stable and not catastrophically slower.

That means the door is open for:
- block-size tuning
- launch tuning
- broader fusion

### H2: numerics compensation may matter as much as raw speed

Because this stack is tuned, a kernel can shift quality even if wallclock is flat.

Likely compensation surfaces:
- `mlp_scale`
- `attn_scale`
- `resid_mix`
- layerwise norm scaling (`ln_scale_factor`)

### H3: full mega-fusion is the real upside, but not the first safe step

The current `triton_act` path is a foothold.

If it stays alive, the next serious kernel target is:
- `linear -> leaky_relu -> square -> linear`

implemented around our banked weight layout.

## Track Policy

- Keep `JR-01` as the untouched winner until a Triton candidate clearly beats it.
- Use this folder for Triton-specific runners, notes, and ablation ordering.
- Move losing Triton variants into `triton/losers/` once decided.
