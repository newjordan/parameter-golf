# Parameter Golf — Synthesized Leaderboard Strategy
**Fractal Int6 QAT + Sliding Window + MLP 3×**

---

## Leaderboard Intelligence (March 19, 2026)

Source: agent-ar.com/leaderboard-param-golf/strategies.html

### Top Techniques by Adoption

| # | Technique | Entries | Impact |
|---|-----------|--------:|--------|
| 1 | Sliding Window Evaluation | 21 | Free 0.01–0.03 BPB at eval time |
| 2 | Int6 Quantization | 18 | 2.67× larger model fits in 16MB |
| 3 | MLP 3× Expansion Ratio | 12 | 25% fewer MLP params per layer → more layers |
| 4 | Quantization-Aware Training (QAT/STE) | 6 | Makes int6 viable (~0.05 BPB saved vs naive PTQ) |
| 5 | Muon Optimizer | 6 | Already in baseline; 1.35× faster convergence |

### Leaderboard State

| Entry | BPB | Δ vs Baseline | Techniques |
|-------|----:|:-------------:|-----------|
| Bhautik Bavadiya (#1) | 0.9695 | ▼ 0.2549 | Unknown — significantly beyond known techniques |
| Sam Larson (#2) | 1.1574 | ▼ 0.0670 | Int6 + MLP 3× + Sliding Window |
| Baseline | 1.2244 | — | Int8 + MLP 2× + fixed-chunk eval |

**Key insight:** Bavadiya's 0.9695 BPB is a **massive** gap (0.1879 BPB ahead of #2).
The known formula of `int6 QAT + MLP 3× + sliding window eval` only gets to ~1.16.
Bavadiya must be using additional techniques beyond what's publicly documented.

---

## Synthesized Strategy: Fractal-Int6-QAT (FIQ)

### Core Thesis

Combine our proven **fractal weight sharing** (7.1% local improvement) with
the **three dominant leaderboard techniques** (int6 QAT, MLP 3×, sliding window).
Weight sharing frees even more parameter budget than the leaderboard entries have,
because shared weights only appear once in the artifact.

### Why This Combination is Stronger Than Either Alone

The leaderboard entries use int6 to fit a ~42MB FP16 model into 16MB.
Our fractal approach lets us train with **fewer unique parameters** that
produce **more effective layers**. Combined with int6 QAT, we can either:

1. **Go wider** — same unique param count, but wider layers with int6 headroom
2. **Go deeper** — more fractal loops with the same unique layers
3. **Both** — int6 buys 2.67× capacity, fractal buys another ~2× via sharing

```
LEADERBOARD LEADER (Bhautik Bavadiya, 0.9695 BPB):
  ~42M params in FP16 → int6 QAT → 16MB artifact
  MLP 3× expansion, standard unique layers

OUR APPROACH:
  N unique layers × K loops = N×K effective layers
  MLP 3× expansion (matching leader)
  Int6 QAT (matching leader)
  Sliding window eval (matching leader)
  + Weight sharing → even bigger effective model in same 16MB
```

---

## The Five Pillars

### 1. Int6 Quantization-Aware Training (from leaderboard #1 & #4)

Replace the baseline's post-training int8 quantization with int6 QAT using STE.

**Why int6 over int8:**
- Int8: 256 levels, ~1× model size (baseline already fits)
- Int6: 64 levels, 2.67× larger model fits in 16MB
- Int4: 16 levels, too much quality loss even with QAT
- Int6 is the competition sweet spot — 18/47 scored entries use it

**Implementation:**

```python
def int6_quantize(w, scale):
    """Fake-quantize weights to 6-bit range [-31, 31]."""
    w_q = torch.clamp(torch.round(w / scale), -31, 31)
    return w_q * scale

class Int6STE(torch.autograd.Function):
    """Straight-Through Estimator for int6 quantization."""
    @staticmethod
    def forward(ctx, w, scale):
        return int6_quantize(w, scale)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: gradient passes through as if quantization wasn't there
        return grad_output, None
```

**Training loop change:**
- Forward pass: quantize all weight matrices to int6 via STE
- Backward pass: gradients flow through unmodified
- Optimizer updates float32 master weights as usual
- Model learns weight distributions that survive 6-bit discretization

**Serialization change:**
- Export 6-bit packed weights (3 weights per 18 bits, or 4 per 3 bytes)
- Per-row scale factors in FP16 (same as baseline's int8 scheme)
- zlib compression on top (shared weights compress exceptionally well)

### 2. MLP 3× Expansion Ratio (from leaderboard #3)

Change MLP hidden dimension from `2 × model_dim` to `3 × model_dim`.

**Why 3× beats 2×:**
- Baseline uses 2× (conservative, small model)
- Traditional Transformers use 4× (too expensive here)
- 3× is the sweet spot for parameter-constrained models
- More MLP capacity per layer = better feature extraction
- The extra cost is offset by int6's 2.67× compression headroom

```python
class MLP(nn.Module):
    def __init__(self, dim, mlp_mult=3):  # Changed from 2 to 3
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.proj(x.square())
```

### 3. Sliding Window Evaluation (from leaderboard #2)

Instead of evaluating on fixed-length chunks, use overlapping windows.

**Impact:** 0.01–0.03 BPB improvement for free (no training change).

```python
def eval_sliding_window(model, tokens, window=1024, stride=512):
    """Score each token with full window of prior context."""
    total_loss = 0
    total_tokens = 0
    for start in range(0, len(tokens) - window, stride):
        chunk = tokens[start:start + window + 1]
        x, y = chunk[:-1], chunk[1:]
        logits = model(x.unsqueeze(0))
        # Only score the last `stride` tokens (they have full context)
        score_start = window - stride
        loss = F.cross_entropy(
            logits[0, score_start:], y[score_start:], reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += stride
    return total_loss / total_tokens
```

### 4. Fractal Weight Sharing (our innovation)

Use `N` unique layers repeated in `K` loops for `N×K` effective depth.

**Proven results (local, 300 steps):**
- Fractal 3×3 at 864d: 2.5953 BPB (7.1% better than baseline 2.7927)
- Weight sharing compresses extremely well under quantization

**For the synthesized strategy, explore:**
- 5×2 = 10 effective layers (wider, more unique layers)
- 3×3 = 9 effective layers (proven locally, fewest unique params)
- 4×3 = 12 effective layers (deepest option)

With int6 QAT, we can afford much wider fractal layers:

```
Fractal 3×3 with int6 QAT:
  3 unique layers, 3 loops = 9 effective layers
  At int6: ~1100d model fits in 16MB (vs 864d at int8)
  That's 2.15× wider than baseline!
```

### 5. Gravity with Warmup (our innovation, refined)

Auxiliary losses at each loop boundary, with a warmup period to avoid
noisy gradients in early training.

```python
# Zero gravity for first 1000 steps, then ramp up over 500 steps
gravity_scale = max(0, min((step - 1000) / 500, 1.0))
```

---

## Combined Architecture

```
INPUT TOKENS (1024 vocab)
    │
    ▼
EMBEDDING (1024 × ~1100d, tied)
    │
    ▼
RMS NORM → x0
    │
    ▼
LOOP 1 (3 unique layers, loop_pos=0):
    ├── Block A: attn(GQA) + MLP(3×, ReLU²)
    ├── Block B: attn(GQA) + MLP(3×, ReLU²)
    ├── Block C: attn(GQA) + MLP(3×, ReLU²)
    ├── GRAVITY: peek → loss₁ (warmed up)
    └── Store output
    │
    ▼
LOOP 2 (same 3 layers, loop_pos=1):
    ├── AttnRes: attend over previous loop outputs
    ├── Block A-C (shared weights, loop_pos=1)
    ├── GRAVITY: peek → loss₂ (warmed up)
    └── Store output
    │
    ▼
LOOP 3 (same 3 layers, loop_pos=2):
    ├── AttnRes: attend over all previous outputs
    ├── Block A-C (shared weights, loop_pos=2)
    └── FINAL LOSS
    │
    ▼
INT6 QAT: fake-quantize in forward, STE in backward
    │
    ▼
EVAL: Sliding window (stride=512, window=1024)
    │
    ▼
SERIALIZE: int6 packed weights + zlib → ≤16MB
```

---

## Parameter Budget (Int6 QAT, 3×3 Fractal at ~1100d)

```
Component                    | Unique Params | Int6+zlib est.
=============================|==============:|===============
Embedding (1024 × 1100)     |    1,126,400  | ~650KB
3 unique blocks:             |               |
  Attn Q (1100 → 1100)      |    1,210,000  |
  Attn K (1100 → 550, GQA)  |      605,000  |
  Attn V (1100 → 550, GQA)  |      605,000  |
  Attn Proj (1100 → 1100)   |    1,210,000  |
  MLP fc (1100 → 3300, 3×)  |    3,630,000  |
  MLP proj (3300 → 1100)    |    3,630,000  |
  Per-block subtotal         |   10,890,000  |
  × 3 blocks                 |   32,670,000  | ~18MB raw → ~6MB int6
Norms, scales, etc.          |      ~33,000  | ~20KB
Loop pos emb (3)             |        3,300  | ~2KB
Gravity weights (3)          |            3  | ~0B
=============================|==============:|===============
TOTAL unique params          |  ~33,832,703  |
Int6 packed (×0.75 bytes)    |              | ~25.4MB raw
zlib compressed              |              | ~13-14MB est.
+ code (~50KB)               |              | ~13-14MB total
```

**This is ~2× more effective parameters than the baseline's 17M, fitting
in the same 16MB artifact.** Weight sharing means the 9-effective-layer
model only stores 3 layers' worth of unique weights.

If 1100d is too tight, fall back to 1024d or 960d.

---

## Hyperparameter Recommendations

```bash
# Architecture
NUM_LAYERS=3          # 3 unique layers
NUM_LOOPS=3           # 3 loops = 9 effective layers
MODEL_DIM=1100        # much wider with int6 headroom
NUM_HEADS=10          # 1100/10 = 110 head_dim (even, for RoPE)
NUM_KV_HEADS=5        # GQA 2:1 ratio
MLP_MULT=3            # 3× expansion (leaderboard standard)

# Int6 QAT
QUANT_BITS=6
QAT_ENABLED=1
STE_ENABLED=1

# Sliding Window Eval
EVAL_WINDOW=1024
EVAL_STRIDE=512       # 2× eval compute, ~0.02 BPB gain

# Optimizer
MATRIX_LR=0.04        # Muon (baseline)
SCALAR_LR=0.04
EMBED_LR=0.6

# Gravity
GRAVITY_WARMUP_STEPS=1000
GRAVITY_RAMP_STEPS=500
```

---

## Experiment Sequence

### Phase 1: Int6 QAT Foundation
1. Add int6 QAT with STE to baseline (9 unique layers, 512d)
2. Increase model_dim until artifact = ~15.5MB
3. Verify BPB improvement from larger model
4. Add MLP 3× expansion, retune dim

### Phase 2: Sliding Window Eval
5. Implement sliding window evaluation
6. Sweep stride: 64, 128, 256, 512
7. Measure BPB gain and eval time tradeoff

### Phase 3: Fractal + Int6
8. Replace 9 unique layers with 3×3 fractal
9. Widen to ~1100d (feasible with int6 + sharing)
10. Compare vs Phase 1 best (unique layers + int6)

### Phase 4: Add Gravity + AttnRes
11. Add gravity with warmup (skip first 1000 steps)
12. Add AttnRes depth attention
13. Full combination: fractal + int6 QAT + MLP 3× + sliding window + gravity

### Phase 5: Submission
14. Verify artifact ≤ 16MB with int6 serialization
15. Run on 8×H100 for 10 minutes
16. Prepare submission folder with logs, README, submission.json

---

## Key Differences From Our Previous PLAN.md

| Aspect | Previous Plan | Synthesized Strategy |
|--------|--------------|---------------------|
| Quantization | Int8 (baseline) | **Int6 QAT with STE** |
| MLP expansion | 2× (baseline) | **3× (leaderboard standard)** |
| Evaluation | Fixed chunks | **Sliding window (stride-512)** |
| Model width | 864d (int8 budget) | **~1100d (int6 budget)** |
| Unique params | ~16.5M | **~33.8M** |
| Effective depth | 9 (3×3) | 9 (3×3) or 10 (5×2) |
| Primary lever | Weight sharing alone | **Weight sharing × int6 compression** |

The combination of fractal weight sharing AND int6 QAT is the key insight:
weight sharing reduces unique parameters, int6 reduces bytes per parameter.
The multiplicative effect lets us pack far more effective model capacity
into 16MB than either technique alone.

---

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| Int6 QAT too noisy for fractal shared weights | Start with int8 fractal as fallback; gradually reduce to int6 |
| 1100d model exceeds 16MB | Fall back to 1024d or 960d; weight sharing gives margin |
| MLP 3× too expensive per step | Int6 reduces memory; 10-min budget has compute slack |
| Sliding window eval too slow | Use stride-512 (only 2× cost); eval is small vs training |
| QAT + gravity interaction | Test QAT alone first, then add gravity |

---

## Expected Performance

| Configuration | Expected BPB | Confidence |
|--------------|-------------|-----------|
| Baseline (int8, 2×, 512d) | 1.2244 | Known |
| + Int6 QAT + bigger model | ~1.19 | Medium |
| + MLP 3× | ~1.18 | Medium |
| + Sliding window eval | ~1.16 | Medium |
| + Fractal 3×3 at ~1100d | ~1.14 | Medium-Low |
| + Gravity + AttnRes | ~1.13 | Low |
| Target: beat Sam Larson (#2) | < 1.1574 | Achievable |
| **Target: beat Bavadiya (#1)** | **< 0.9695** | Ambitious stretch goal |

The multiplicative benefit of fractal + int6 is our unique edge.
No leaderboard entry currently uses weight sharing with int6 QAT.

**Note:** Bavadiya's 0.9695 is far beyond what these techniques alone explain.
To truly compete for #1, we likely need architectural innovations beyond int6+fractal.

---

## References

### From Leaderboard Strategies Page
- Nagel et al. (2021) — "A White Paper on Neural Network Quantization" (PTQ + QAT)
- Fan et al. (2020, Meta) — "Training with Quantization Noise for Extreme Compression"
- Bengio et al. (2013) — "Estimating or Propagating Gradients Through Stochastic Neurons" (STE)
- Vaswani et al. (2017) — "Attention Is All You Need" (4× FFN convention)
- Hoffmann et al. (2022) — "Chinchilla" (optimal architecture ratios)
- Shazeer (2020) — "GLU Variants Improve Transformer" (SwiGLU, 2/3× ratio)
- Jordan et al. (2024) — "Muon: An optimizer for hidden layers" (Newton-Schulz orthogonalization)
- Bernstein (2025) — "Deriving Muon" (modular duality theory)

### From Our Previous Work
- Moonshot (2025) — arxiv:2603.15031, Attention Residuals
- Karpathy — autoresearch framework

---

*Strategy synthesized from agent-ar.com leaderboard analysis + local experiments*
*March 19, 2026*
