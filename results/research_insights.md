# Research Insights — Parameter Golf Night 2
**Date:** 2026-03-20
**Author:** Frosty40

## Key Finding: FP16 Embeddings Are the Most Impactful Technique

With a 1024-token vocabulary, keeping the embedding table in float16 instead of int6 quantization is worth **~0.1 BPB** — more than SmearGate, BigramHash, OrthoInit, SWA, and MuonWD combined.

### Evidence

| Config | Quant BPB | Delta |
|--------|-----------|-------|
| MLP3x + earlyQAT + stride64 (no FP16 embed) | 1.2686 | baseline |
| MLP3x + earlyQAT + stride64 + FP16 embed | 1.1693 | **-0.099** |
| + SmearGate + BigramHash + OrthoInit (no FP16) | not tested | — |
| + SmearGate + BigramHash + OrthoInit + FP16 | 1.1725 | -0.096 |

FP16 embed alone accounts for ~93% of the improvement from baseline to our best score.

### Why

The embedding table is the model's entire interface to language. With tied embeddings, it serves as both input lookup and output projection. Int6 quantization (64 discrete levels per dimension) destroys the model's ability to distinguish between similar tokens. Attention and MLP weights tolerate int6 because they operate on continuous hidden states with redundancy from residual connections. The embedding has no such safety net — each token gets exactly one vector.

### Underappreciated on the leaderboard

Most top entries either use FP16 embed without highlighting it, or don't use it at all. The technique appears in PR #60 (@notapplica, 1.1748) but is listed alongside 4 other techniques with no ablation showing its outsized impact. Our controlled A/B test (identical config, only FP16 embed toggled) proves it's the dominant factor.

### Tradeoff

FP16 embed costs ~600KB of artifact space (1024 × 512 × 2 bytes vs ~393KB in int6). This forces shrinking the model elsewhere (we reduced MLP from 1536 to 1344 hidden dim). The BPB tradeoff is overwhelmingly positive — 0.1 BPB gained vs ~0.003 BPB lost from smaller MLP.

## Secondary Finding: Stride=64 Only Works WITH FP16 Embed

Sliding window eval at stride=64 gave **zero improvement** without FP16 embed (1.2686 → 1.2686). But with FP16 embed, stride=64 contributed significantly to the final score. The bottleneck isn't context — it's embedding precision. Fix the embedding first, then stride=64 can exploit the richer representations.

## Experimental Finding: Progressive Fractal Loop Unrolling

Novel technique: start fractal training with 1 loop (fast, ~42ms/step), progressively add loops at step 3000 (2 loops) and step 5000 (3 loops). Rationale: learn language basics fast, then add refinement depth when weights are ready.

**Result:** Got 9086 steps vs 7137 without progressive (27% more steps). But loop transitions caused val BPB instability (bounced from 1.45 to 2.00 during transitions). Pre-quant 1.2706 — not competitive with non-fractal (1.2022). The idea has merit but needs smoother transitions (gradual loop blending instead of hard cutover).

## Tonight's Journey: 1.33 → 1.17

| Step | Quant BPB | What changed |
|------|-----------|-------------|
| Baseline | 1.3306 | 9L/512d MLP2x, int6, no slide |
| + Sliding window | 1.3112 | stride=512, free -0.019 |
| + MLP 3× | 1.2681 | wider FFN, fill 16MB budget |
| + Early QAT 25% | 1.2607 | more fake-quant exposure |
| + FP16 embed + stride64 | 1.1693 | the big one (but over 16MB) |
| + SmearGate + BigramHash + OrthoInit (sized to fit) | **1.1725** | **The Stinky Frost Recipe — PR #190** |
