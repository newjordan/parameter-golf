# Junkyard_Rat_MLP — Full MLP Fusion Branch

Date: 2026-03-29

## Mission
Fork the current `JR-01` winner into a dedicated branch for full MLP mega-fusion work.

This workspace is not for more loader screening.
This workspace is not for activation-only Triton tuning.
This workspace is for the next real systems shot:
- fuse the whole MLP body
- measure whether that buys throughput or BPB
- only then do numerics compensation around the fused branch

## Base Branch

This branch starts from the current best `Junkyard_Rat` path:

- `JR-01` coprime loader winner
- same base stack
- same data path
- same legal discipline

Reference numbers:

| Variant | Step avg | Post-EMA BPB | Sliding BPB |
|---|---:|---:|---:|
| `JR-01` | `91.00ms` | `1.1340` | `1.11056240` |
| `TR-01` `triton_act` | `91.11ms` | `1.1345` | `1.11099954` |
| `TR-02` best delta | `attn_scale=1.02` | `1.1347` cap-time | pending full final eval |

## Fusion Target

Current inner MLP body:

1. `F.linear(h, up_w)`
2. `leaky_relu(0.5)`
3. `square`
4. `F.linear(act, down_w)`

Target:
- fuse as much of that body as possible into one kernel family
- keep `mlp_norm`, `mlp_scale`, and residual add outside in the first pass
- optimize for the real H100 workload, not a toy shape

## Work Order

### Phase 1
- scaffold branch and runners
- preserve the winning base untouched

### Phase 2
- implement fused-MLP microbench
- verify correctness vs eager

### Phase 3
- integrate fused MLP into training
- run full `600s` confirmation

### Phase 4
- if alive, tune numerics around the fused branch
- first target: `attn_scale`

## Non-Goals

Do not use this branch for:
- loader experiments
- activation-only Triton tweaks
- pop-test churn on already-losing deltas

Keep fixed:
- best Phase A/B winner

Decision rule:
- prioritize honest submission readiness over theoretical quant elegance

### Phase D: hand-off to compact architecture line

Only after a base winner exists:
- test whether winning loader/Triton ideas transfer to Bandit_Wagon or crawler-only rebuild

This is where Bandit stays important:
- not as the strongest pure base today
- but as the compact architecture that can spend newly unlocked quality much more efficiently

## Current Comparative Position

| System | Pure base sliding BPB | Combined BPB | Size | Role |
|---|---:|---:|---:|---|
| Rat Rod Green v1 | `1.1129` | `0.4489` | larger | strongest honest base anchor |
| X-WING Cubric | `1.1199` | `0.4820` | `15.58 MB` | strong flat full-budget stack |
| Bandit | `1.1867` | `0.4961` | `~9.2 MB` | compact architecture hedge |
| Medusa_VII DN=0 | `1.1823` | n/a | `9.08 MB` | honest crawler baseline |

## Concrete Read on Bandit

Bandit should be interpreted as:
- a validated compact architecture
- a proof that there is real value in the crawler family even before spending the full 16 MB
- a headroom engine for later width/depth upgrades

Bandit should **not** be interpreted as:
- proof that the crawler base is already better than Rat Rod

So the right organizational split is:
- `Junkyard_Rat`: best clean base-model garage
- `Bandit_Wagon`: best compact-architecture headroom garage

## Exit Criteria

This garage is successful if it produces any one of these:

1. a reproducible base-model run better than `1.1129` sliding
2. an artifact-ready legal run that closes the base-to-roundtrip gap cleanly
3. a systems improvement that clearly transfers to Bandit_Wagon later

## Immediate Next Ablations

1. `JR-02`: `JR-01` winner + fused Triton LeakyReLU^2 MLP
2. `JR-03`: best loader/systems winner + legal Full GPTQ path
3. transfer winning systems ideas into Pocket_Bandit and Shroud_Crawler later

## References

- `experiments/Rat_Rod/PROGRESS.md`
- `experiments/Bandit_Wagon/HYPOTHESIS.md`
- `experiments/Medusa_VII/HYPOTHESIS.md`
- `records/track_10min_16mb/2026-03-29_Bandit_ClownCar_X_CubricNgram9_8xH100/README.md`
- `records/track_10min_16mb/2026-03-26_XWING_Cubric3D_complementary_8xH100/README.md`
