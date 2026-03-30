# Crawler Leg 2 — Ablation: loops=3 + mlp=5.0 combined (1×H100)

## Goal

Combine the two Leg 1 wins (CRAWLER_LOOPS=3 and CRAWLER_MLP_MULT=5.0) into a
single ablation arm to confirm they're **additive**. If they are, this becomes
the new crawler baseline for all future runs.

Then stack LOOP_AWARE_GPTQ=1 + COMPILE_FULLGRAPH=1 from Ablations_v1
(−0.066 combined) on top. That's the full optimized crawler config before
scaling to 8×H100.

## Expected

~−0.15 to −0.18 BPB on int6 sliding window vs Leg 1 baseline.

## Arms

| Arm | Config | Purpose |
|-----|--------|---------|
| CL2-00 | Baseline: loops=4, mlp=4.0 (Leg 1 baseline) | Reference point |
| CL2-01 | loops=3 only | Reproduce Leg 1 single win |
| CL2-02 | mlp=5.0 only | Reproduce Leg 1 single win |
| CL2-03 | **loops=3 + mlp=5.0** | Additivity test (key arm) |
| CL2-04 | loops=3 + mlp=5.0 + LOOP_AWARE_GPTQ=1 | Stack GPTQ |
| CL2-05 | loops=3 + mlp=5.0 + COMPILE_FULLGRAPH=1 | Stack compile |
| CL2-06 | loops=3 + mlp=5.0 + LOOP_AWARE_GPTQ=1 + COMPILE_FULLGRAPH=1 | Full optimized config |

## Compute

- 1×H100, 600s/arm, 7 arms = ~70 min total
- SKIP_EMA=1, SKIP_GPTQ=1 for speed (except arms testing GPTQ)
- Ngram eval disabled (NGRAM_EVAL_ORDER=0) for pure base model comparison

## Architecture (from Bandit)

- 4 flat + 1 crawler × N loops, inst_dim=32 FLOW
- DN=0 (DeltaNet off — causality fix)
- CRAWLER_QUANT_INT8=1
- Medusa harness (Bandit copy)

## Key Questions

1. Are loops=3 and mlp=5.0 wins additive? (CL2-03 delta ≈ CL2-01 delta + CL2-02 delta?)
2. Does LOOP_AWARE_GPTQ provide additional gain on top of the combined config?
3. Does COMPILE_FULLGRAPH improve throughput → more steps → better val_bpb?
4. Is the full stack (CL2-06) strictly better than partial stacks?

## Results

| Arm | Label | Steps | val_bpb | Delta vs CL2-00 |
|-----|-------|-------|---------|------------------|
| CL2-00 | baseline | — | — | — |
| CL2-01 | loops=3 | — | — | — |
| CL2-02 | mlp=5.0 | — | — | — |
| CL2-03 | loops=3+mlp=5.0 | — | — | — |
| CL2-04 | +GPTQ | — | — | — |
| CL2-05 | +compile | — | — | — |
| CL2-06 | full stack | — | — | — |
