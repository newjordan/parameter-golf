# Phase 1 — Baseline Optimization Program

## Objective
Optimize `train_gpt.py` to minimize **val_bpb** (bits-per-byte on the FineWeb validation set) while keeping the compressed artifact under **16MB** and training within **10 minutes on 8×H100 GPUs**.

**Current baseline:** 1.2244 BPB (post-quantization roundtrip)
**Target:** ~1.18 BPB (replicate ArjunAutoResearch-class result)
**Expected:** 8–15 genuine improvements from ~100 experiments

## Constraints
- **Artifact size:** code + int8+zlib compressed weights ≤ 16,000,000 bytes
- **Training time:** ≤ 600 seconds wallclock on 8×H100
- **No external downloads** during evaluation
- **Reproducibility:** all changes must be deterministic given the same seed

## How the Script Works
All hyperparameters in `train_gpt.py` are controlled via environment variables (see the `Hyperparameters` class, lines 39–87). You do NOT need to edit the Python code for Phase 1 — just set env vars before launching.

### Key Defaults (Baseline)
| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| Matrix LR | `MATRIX_LR` | 0.04 | Muon optimizer LR for weight matrices |
| Scalar LR | `SCALAR_LR` | 0.04 | Adam LR for vectors/scalars |
| Tied Embed LR | `TIED_EMBED_LR` | 0.05 | Adam LR for tied embedding table |
| Embed LR | `EMBED_LR` | 0.6 | Adam LR for untied embedding (unused when tied) |
| Head LR | `HEAD_LR` | 0.008 | Adam LR for untied lm_head (unused when tied) |
| Train batch tokens | `TRAIN_BATCH_TOKENS` | 524288 | Tokens per training step (across all GPUs) |
| Train seq len | `TRAIN_SEQ_LEN` | 1024 | Sequence length for training |
| Warmup steps | `WARMUP_STEPS` | 20 | Torch compile warmup (state restored after) |
| Warmdown iters | `WARMDOWN_ITERS` | 1200 | LR decay window before wallclock cap |
| Muon momentum | `MUON_MOMENTUM` | 0.95 | Final Muon momentum |
| Muon momentum warmup start | `MUON_MOMENTUM_WARMUP_START` | 0.85 | Initial Muon momentum |
| Muon momentum warmup steps | `MUON_MOMENTUM_WARMUP_STEPS` | 500 | Steps to ramp momentum |
| Muon backend steps | `MUON_BACKEND_STEPS` | 5 | Newton-Schulz iterations |
| Beta1 | `BETA1` | 0.9 | Adam beta1 |
| Beta2 | `BETA2` | 0.95 | Adam beta2 |
| Grad clip norm | `GRAD_CLIP_NORM` | 0.0 | Gradient clipping (0 = disabled) |
| QK gain init | `QK_GAIN_INIT` | 1.5 | Per-head Q gain initialization |
| Logit softcap | `LOGIT_SOFTCAP` | 30.0 | Logit capping value |
| Rope base | `ROPE_BASE` | 10000.0 | RoPE frequency base |
| Model dim | `MODEL_DIM` | 512 | Hidden dimension |
| Num layers | `NUM_LAYERS` | 9 | Transformer layers |
| Num heads | `NUM_HEADS` | 8 | Query attention heads |
| Num KV heads | `NUM_KV_HEADS` | 4 | Key-value heads (GQA) |
| MLP mult | `MLP_MULT` | 2 | MLP hidden dim multiplier |
| Tied embeddings | `TIE_EMBEDDINGS` | 1 | Tie input/output embeddings |
| Val loss every | `VAL_LOSS_EVERY` | 1000 | Validation frequency |
| Iterations | `ITERATIONS` | 20000 | Max training iterations |
| Max wallclock | `MAX_WALLCLOCK_SECONDS` | 600.0 | Training time cap |
| Seed | `SEED` | 1337 | Random seed |

## Phase 1 Experiment Categories

### Category 1: Learning Rate Tuning (highest priority)
Sweep the key learning rates one at a time, then jointly:
- `MATRIX_LR`: [0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.08]
- `SCALAR_LR`: [0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
- `TIED_EMBED_LR`: [0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
- After finding best individuals, do a joint 3×3×3 grid around the winners

### Category 2: Batch Size Optimization
- `TRAIN_BATCH_TOKENS`: [262144, 393216, 524288, 786432, 1048576]
- Larger batches = fewer steps in 10 min but better gradient estimates
- Smaller batches = more steps, more noise, potentially better generalization

### Category 3: Warmup & Warmdown Schedule
- `WARMDOWN_ITERS`: [800, 1000, 1200, 1500, 2000, 2500, 3000]
- `MUON_MOMENTUM`: [0.90, 0.93, 0.95, 0.97]
- `MUON_MOMENTUM_WARMUP_START`: [0.80, 0.85, 0.90]
- `MUON_MOMENTUM_WARMUP_STEPS`: [200, 500, 800, 1000]

### Category 4: Regularization & Gradient Control
- `GRAD_CLIP_NORM`: [0.0, 0.5, 1.0, 2.0, 5.0]
- `BETA1`: [0.85, 0.9, 0.95]
- `BETA2`: [0.90, 0.95, 0.98, 0.99]

### Category 5: Architecture-Adjacent Tweaks
- `QK_GAIN_INIT`: [1.0, 1.25, 1.5, 1.75, 2.0]
- `LOGIT_SOFTCAP`: [20.0, 25.0, 30.0, 40.0, 50.0]
- `ROPE_BASE`: [5000, 10000, 50000, 100000]
- `TRAIN_SEQ_LEN`: [512, 768, 1024, 1536, 2048]

### Category 6: Model Shape (within 16MB budget)
- Wider+shallower: `NUM_LAYERS=7 MODEL_DIM=576`
- Deeper+narrower: `NUM_LAYERS=11 MODEL_DIM=448`
- More heads: `NUM_HEADS=16 NUM_KV_HEADS=4`
- Different GQA: `NUM_HEADS=8 NUM_KV_HEADS=2` or `NUM_KV_HEADS=8`
- MLP ratio: `MLP_MULT=3` (with smaller dim to fit budget)

## Experiment Protocol

### For each experiment:
1. Set the relevant env vars
2. Run training to completion (10 min cap or convergence)
3. Record: `val_bpb` (pre-quant), `val_bpb` (post-quant roundtrip), artifact size, training time, steps completed
4. Log results to `experiments/results.jsonl`

### Decision rules:
- An improvement is **genuine** if post-quant val_bpb improves by ≥ 0.001 over the current best
- Always validate with the full roundtrip (int8+zlib decompress → eval)
- If artifact exceeds 16MB, the experiment is invalid regardless of BPB
- Re-run the best config 3× with different seeds to confirm stability

### Iteration strategy:
1. Start with single-variable sweeps (Categories 1–5)
2. Combine the best settings from each category
3. Fine-tune the combined config with narrow sweeps
4. Test architecture changes (Category 6) with the best hyperparameters
5. Final validation: 3 seeds on the champion config

## Output Format
Each experiment writes a line to `experiments/results.jsonl`:
```json
{
  "experiment_id": "lr_matrix_0.05",
  "timestamp": "2026-03-19T12:00:00Z",
  "env_vars": {"MATRIX_LR": "0.05"},
  "val_bpb_prequant": 1.2172,
  "val_bpb_postquant": 1.2244,
  "artifact_bytes": 15863489,
  "training_time_ms": 600038,
  "steps_completed": 13780,
  "hardware": "8xH100",
  "notes": "baseline comparison"
}
```

## Hardware Configurations

### H100 (Official Leaderboard)
- 8×H100 80GB HBM3
- `torchrun --nproc_per_node=8 train_gpt.py`
- `MAX_WALLCLOCK_SECONDS=600`
- This is the config that matters for submission

### DGX Spark / H10 (Research Proxy)
- 1× GPU, reduced compute
- `python train_gpt.py` (single GPU, no torchrun)
- `MAX_WALLCLOCK_SECONDS=1800` (30 min for comparable token coverage)
- `ITERATIONS=50000` (more steps at lower throughput)
- Results are directional — BPB numbers will differ but relative ordering should transfer
- Use for cheap hypothesis testing before burning H100 time
