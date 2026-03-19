# 🏌️ Parameter Golf Council Report
## OpenAI Model Craft Challenge + Autoresearch Strategy Analysis

**Council Date:** 2026-03-19
**Brains:** Codex 5.3 Spark (Quant), Opus 4.6 (Risk), Venice/Spark (Synthesis)

---

## Competition Summary

- **Goal:** Train the best LM that fits in **16MB artifact** (code + int8+zlib compressed model)
- **Training budget:** 10 min on 8×H100s
- **Metric:** Bits per byte (BPB) on FineWeb validation set — lower is better
- **Baseline SOTA:** 1.2244 BPB (naive), 1.2074 BPB (4-hour unlimited)
- **Must beat by:** ≥0.005 nats for SOTA record
- **Runs:** March 18 → April 30, 2026

---

## Budget Math (Critical)

The 16MB is **not** 16MB of parameters. It's the **total artifact**:
- Code (train_gpt.py + dependencies) ~47KB
- Compressed model (int8 quantized + zlib)
- Baseline compressed model: ~15.8MB
- **Headroom: ~150KB** for code growth

**Effective param capacity:**
- Raw int8: 1 byte/param → ~16M params max
- Zlib on int8 weights: ~60-70% ratio → **~22-25M effective params**
- Baseline uses ~20-25M params compressed into budget
- **You're already near ceiling with the baseline architecture**

---

## TIER 1: HIGH-CONFIDENCE STRATEGIES (Do These First)

### 1. Depth Recurrence (Weight-Tied Layers) ⭐⭐⭐
**Council consensus: STRONGEST single strategy**

| Metric | Value |
|--------|-------|
| Expected BPB gain | -0.005 to -0.012 |
| Implementation complexity | 2/5 |
| 16MB risk | LOW (reduces params, helps compression) |
| Autoresearch-friendly | YES |

**How:** Share weights across N layers, loop K times (e.g., 3 unique layers × 6 loops = 18 effective depth). Freed param budget → wider dims or bigger vocab.

**Risk brain warning:** ALBERT showed weight-shared transformers underperform at small scales vs distinct-layer models. The compression win may not offset quality loss. At 512-dim, each recurrence pass has diminishing representational gain because the residual stream is narrow.

**Synthesis verdict:** Try it, but don't commit to full sharing. **Partial recurrence** (share FFN weights but keep unique attention) is the sweet spot. Start with 2 unique layer groups × 4-5 loops.

### 2. Training Optimization (LR + Optimizer + Schedule)
**Council consensus: Cheapest improvement, do immediately**

| Metric | Value |
|--------|-------|
| Expected BPB gain | -0.003 to -0.008 |
| Implementation complexity | 1/5 |
| 16MB risk | NONE |
| Autoresearch-friendly | EXTREMELY YES |

**How:**
- **Warmup + cosine annealing** with aggressive decay — the baseline likely undertunes this for 10min
- **Muon optimizer** (already in autoresearch train.py) — tune β, LR independently for embed vs body
- **Higher initial LR** — the 10min budget means you want to learn fast and decay hard
- **Batch size scheduling** — start small batches (faster steps), grow mid-training for stability

**Autoresearch angle:** This is *the* strategy for program.md. The agent can sweep LR/schedule/batch combos automatically at 12 experiments/hour and find the local optimum overnight.

### 3. Evaluation Sequence Length Optimization ⭐⭐
**Council consensus: Free BPB, almost nobody will try first**

| Metric | Value |
|--------|-------|
| Expected BPB gain | -0.003 to -0.010 |
| Implementation complexity | 1/5 |
| 16MB risk | NONE |
| Autoresearch-friendly | YES (but needs manual eval experiments) |

**How:** The rules say "we allow evaluation at any sequence length." Longer eval sequences give more context → better predictions → lower BPB. The baseline trains at 1024 seqlen. **Evaluate at 2048 or 4096** even if you trained at 1024. The model can still attend to longer context if you use RoPE/ALiBi that extrapolates.

**Risk brain warning:** This only works if the model was trained with a position encoding that extrapolates (RoPE does, learned absolute PE does not). Also, longer eval = slower eval, and there's a 10-min eval time cap.

**Synthesis verdict:** Almost certainly worth doing. Switch to RoPE if not already using it, then eval at max feasible length.

---

## TIER 2: MEDIUM-CONFIDENCE STRATEGIES

### 4. Quantization-Aware Training (QAT)
| Expected BPB gain | -0.002 to -0.005 (indirect — more params in budget) |
| Complexity | 3/5 |
| Autoresearch-friendly | PARTIAL |

**How:** Simulate int8 quantization in forward pass during training. The model learns to be robust to quantization, so post-training int8 loses less quality. The real win: if QAT + aggressive zlib achieves 0.5 bytes/param, your budget grows to ~33M params.

**Risk:** QAT adds training overhead (~20% slower per step), eating into your 10-min budget. The net param gain needs to compensate.

### 5. Wider + Shallower + Recurrence
| Expected BPB gain | -0.003 to -0.008 |
| Complexity | 1/5 |
| Autoresearch-friendly | YES |

**How:** 4 layers × 768-dim instead of 9 × 512, with 3× recurrence loop. Wider layers capture more per-position info. The compression ratio of 4 unique layers is better than 9 unique layers.

### 6. Custom Tokenizer / Vocab Size Tuning
| Expected BPB gain | -0.002 to -0.006 |
| Complexity | 3/5 |
| Autoresearch-friendly | NO (one-time manual experiment) |

**How:** The metric is BPB (tokenizer-agnostic). A vocab of 1024 is very small. Experimenting with 512 or 2048 could shift the compression/representation tradeoff. Smaller vocab = smaller embedding table = more params for transformer body. But smaller vocab = more tokens per document = need more steps.

**Risk brain note:** Byte-level (vocab=256) is theoretically cleanest but requires much deeper model to compensate. The 10-min training budget likely kills this.

---

## TIER 3: HIGH-RISK / HIGH-REWARD

### 7. Test-Time Compute (Evaluation-Phase Tricks)
| Expected BPB gain | -0.005 to -0.020 (if it works) |
| Complexity | 4/5 |
| Autoresearch-friendly | NO |

**How:** Multiple forward passes at eval time — e.g., sliding window with overlap, or self-consistency sampling. The 10-min eval budget gives room for creative inference strategies.

**Risk brain warning:** The rules are ambiguous on what "evaluation" means. If BPB is computed as log-loss per byte, you can't beam-search your way to better BPB — you need the actual next-token probability. Self-consistency doesn't apply to perplexity evaluation. **Most test-time compute tricks don't help BPB.** The exception: adaptive context length (see strategy 3).

### 8. Mixture of Experts
| Expected BPB gain | -0.008 to -0.015 |
| Complexity | 4/5 |
| 16MB risk | HIGH |
| Autoresearch-friendly | PARTIAL |

**Risk brain warning:** MoE at 512-dim is almost certainly a trap. Router overhead eats param budget at this scale. Load balancing during training may not transfer to validation distribution. Code complexity adds bytes.

### 9. Test-Time Training (TTT)
| Expected BPB gain | UNKNOWN (potentially large) |
| Complexity | 5/5 |
| Disqualification risk | MEDIUM |

**How:** Fine-tune the model on the test sequence itself during evaluation. This is explicitly interesting to the competition organizers ("test-time training" is listed as an example of creative submissions).

**Risk:** What counts as "external compute"? If you're doing gradient steps during eval, that's compute you're not counting in training time. The rules say 10-min eval cap, so you get up to 10 minutes of TTT. But this feels like it might violate the spirit of the challenge.

---

## AUTORESEARCH IMPLEMENTATION STRATEGY

### How to Structure program.md for Maximum Discovery

**Phase 1 (Hours 1-8): Hyperparameter sweep**
- Agent sweeps LR, batch size, warmup steps, weight decay
- Each experiment is 5 min — fits 12/hour
- Expected: find 0.003-0.005 BPB improvement from training config alone

**Phase 2 (Hours 8-16): Architecture search**
- Agent tries depth/width/recurrence combos
- Key knobs: NUM_LAYERS, MODEL_DIM, num_recurrence_loops, NUM_KV_HEADS
- Expected: find another 0.003-0.008 from architecture

**Phase 3 (Hours 16-24): Compression + eval tricks**
- Agent experiments with eval sequence length
- Agent tries partial weight sharing patterns
- Expected: find another 0.002-0.005

**program.md template:**
```
1. Record current best val_bpb
2. Choose ONE modification from the priority list
3. Implement it in train.py
4. Train for 5 minutes
5. Compare val_bpb to best
6. If better: keep changes, update best
7. If worse: revert
8. Log the experiment result
9. Repeat with next modification

Priority list (try in order):
- Learning rate: try 2x, 0.5x, 3x current
- Batch size: try 2x, 0.5x
- Warmup steps: try 0, 100, 500, 1000
- Weight decay: try 0.01, 0.05, 0.1, 0.2
- Cosine schedule: try different min_lr ratios
- NUM_LAYERS: try 6, 12, 15
- MODEL_DIM: try 384, 640, 768
- Recurrence: add weight tying with 2x, 3x, 4x loops
- Eval seqlen: try 2048, 4096
```

### Autoresearch Traps (from Risk Brain)
- **Looks automatable but isn't:** MoE routing, SSM kernels, custom tokenizers, test-time training
- **Looks hard but IS automatable:** depth recurrence (it's literally one for-loop change), LR/schedule sweeps, width/depth tradeoffs

---

## THEORETICAL FLOOR

**What does information theory say about 16MB / FineWeb BPB?**

- FineWeb is web text with estimated entropy ~0.8-1.0 BPB for an ideal compressor
- A 16MB model has ~log₂(256^16M) ≈ 128M bits of information capacity
- FineWeb validation is ~50K documents — the model needs to generalize, not memorize
- **Realistic floor for 16MB:** ~1.10-1.15 BPB (leaving ~0.07-0.12 to gain from baseline)
- **Theoretical perfect compressor floor:** ~0.8-0.9 BPB (unreachable at this scale)

---

## FINAL COUNCIL SYNTHESIS

### Recommendation: Layered Attack Plan

**Immediate (Day 1):**
1. Clone parameter-golf, set up autoresearch
2. Run hyperparameter sweep overnight (Phase 1 program.md)
3. Manually test eval at longer sequence lengths

**Week 1:**
4. Implement partial depth recurrence (share FFN, unique attention)
5. Run autoresearch on architecture width/depth combos
6. Implement QAT if param headroom is the bottleneck

**Week 2-3:**
7. Explore tokenizer size variants (manual experiments)
8. Consider test-time training if rules permit
9. Submit best result, iterate

### Confidence Score: **72/100**
*We can beat the baseline. Breaking 1.20 BPB is realistic. Breaking 1.15 is hard. Breaking 1.10 requires a breakthrough.*

### Agreement Score: **85/100**
*Both brains agree on: depth recurrence + training optimization + eval tricks as the core attack. Disagreement on: MoE viability (quant brain more optimistic than risk brain) and test-time training legality.*

### Key Disagreements
1. **MoE:** Quant brain sees -0.008 to -0.015 potential. Risk brain calls it "almost certainly a trap at 512-dim." **Synthesis: avoid MoE unless you have >1 week to debug routing.**
2. **Depth recurrence quality:** Quant brain estimates -0.005 to -0.012. Risk brain warns ALBERT showed underperformance at small scales. **Synthesis: partial sharing (FFN only) mitigates this.**
3. **Test-time training:** Both see potential but uncertain legality. **Synthesis: ask in the Discord #parameter-golf-discussions before investing time.**

### Top 3 Action Items
1. **Tonight:** Set up autoresearch with Phase 1 hyperparameter program.md, let it run 100 experiments overnight
2. **Tomorrow:** Implement partial depth recurrence + longer eval sequence length
3. **This week:** Submit first improved result, then iterate with QAT and architecture search

---

## APPENDIX: Exa Deep Research Findings (2026-03-19)

### Key Papers & Techniques Discovered

**1. Relaxed Recursive Transformers (Bae et al., 2024)**
- URL: <https://arxiv.org/abs/2410.20672>
- Layer-wise LoRA on top of weight sharing — adds tiny per-layer adapters to shared weights
- This is exactly the "partial recurrence" strategy the council recommended
- **Direct applicability: HIGH** — implement shared layers + per-loop LoRA adapters

**2. Mixture-of-Recursions (MoR) (Bae et al., 2025)**
- URL: <https://arxiv.org/html/2507.10524v3>
- Adaptive per-token recursion depth with lightweight routers
- Combines parameter sharing with adaptive compute
- "Forms a new Pareto frontier: at equal training FLOPs and smaller model sizes, significantly lowers validation perplexity"
- **Direct applicability: HIGH** — but implementation complexity is 4/5

**3. DeltaLLM (Mikaelyan et al., 2025)**
- URL: <https://arxiv.org/pdf/2501.18596>
- Low-rank deltas between shared weights — 12% param reduction retaining 90% performance
- Outperforms SliceGPT, ShortGPT, LaCo at same compression
- **Direct applicability: MEDIUM** — post-training technique, but concept applies to training

**4. Entropy-Weighted Quantization (EWQ) (Behtash et al., 2025)**
- URL: <https://arxiv.org/html/2503.04704v1>
- Selective per-layer quantization based on entropy distribution
- **Surprising finding: can REDUCE perplexity vs unquantized models** (regularization effect)
- Up to 18% memory reduction while maintaining MMLU within 0.5%
- **Direct applicability: HIGH** — apply entropy-based selective quant instead of uniform int8

**5. Compute-Optimal QAT (Dremov et al., 2026)**
- URL: <https://arxiv.org/pdf/2509.22935>
- Optimal ratio of QAT to full-precision training increases with total compute
- **Novel: cooldown + QAT fusion** eliminates redundant FP updates
- **Direct applicability: HIGH** — fuse cosine cooldown with QAT phase

**6. PyTorch QAT Results (2024)**
- URL: <https://pytorch.org/blog/quantization-aware-training/>
- QAT on Llama3-8B: recovers 96% hellaswag accuracy degradation, 68% perplexity degradation
- After XNNPACK lowering: QAT model = 16.8% lower perplexity than PTQ, same size
- BPB: 0.836 (QAT) vs 0.887 (PTQ) — **0.051 BPB improvement from QAT alone**

**7. Autoresearch Proven Results**
- Autoresearch already cut nanochat Time-to-GPT-2 speedrun from 2.02h to 1.80h on 8xH100
- 333 experiments run in one night by 35 agents on Hyperspace network
- Agent found improvements across architecture, hyperparameters, and optimizer that humans missed

### Updated Strategy Priority (Post-Exa)

The Exa findings strengthen several strategies:
1. **Relaxed Recursive Transformers + LoRA** → upgraded from "try partial recurrence" to "implement exactly this paper"
2. **Entropy-Weighted Quantization** → NEW strategy not in original council report — potentially free BPB from selective quant
3. **QAT cooldown fusion** → specific technique to save compute budget while getting QAT benefits
4. **MoR (adaptive recursion depth)** → high potential but complex; save for week 2
