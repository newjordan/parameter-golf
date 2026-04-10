# Crawler Science Board

Track: Bandit_Wagon lineage · Goal: best int6_sw_bpb + smallest artifact ≤ 16MB
Champion: **1.13867894 BPB** (seed 444) · **15,239,617 bytes** · `records/track_10min_16mb/2026-04-02_Bandit_Wagon_X_9F_8xH100/`

Legend: → PROMOTED · ✓ PASS · ✗ FAIL · ⏳ PENDING · — n/a

---

## Status Sync (2026-04-10)

- In-tree promotion baseline is now TK1 recovered legal 7F+3C full run trio: seed `444=1.13541288`, seed `300=1.13853446`, seed `4=1.13536063`, mean `1.13643599`, max artifact `15,902,698` bytes.
- **Trapper Keeper 1 (8F+3C)** production gate: `1.13526829` int6_sw_bpb — beats leader by `-0.00341`. BUT artifact `17,948,983` bytes (over 16MB cap with zstd). Brotli recompress pending. Pod lacked FA3 (157ms/step, 3811 steps). With FA3+more steps, quality would likely improve further.
- **TK1 recovered 7F+3C + GPTQ + pyminify + selective prune** is now fully confirmed on seeds 444/300/4 and is the current in-tree crawler winner.
- Layer relationship grid (5×4, 20 arms) mapped full flat×crawler surface. 8F+3C is quality peak. 3C matches 3-loop symmetry. 4C reverses. Ridge runs 8F+3C → 7F+4C → 7F+3C.
- Corpus ablation validated: multi-crawler (-0.0119), 4-loop diff battery (-0.0046), anchor (-0.0024), QAT softclamp (-0.0047). Dead: INST_DIM=64, sigmoidste, C>loops symmetry.
- Safe vs aggressive (3 vs 4 loops on 8F+3C): 3 loops wins on wallclock. 4th loop is waste.
- Helix_ab_3 gate result is a hard fail: `+0.13954768` int6_sw_bpb vs control.
- Ouroboros PR lineage exists externally (`#1283` / `#1308`) but not in-tree.

---

## Thread: Baseline — Crawler loops=3 / mlp=6.0

Established in Leg 3. This is the root config all BW legs descend from.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| 2026-03-29 | Leg 3 (CL3-01) | loops=3, mlp=6.0, SKIP_GPTQ=1, 600s wallclock | — | — | 1.18720 | 8.84MB | → PROMOTED | Extra time > GPTQ tricks. Quant gap nearly closed. 7MB headroom. |
| 2026-03-29 | BW4 | unknown | — | — | 1.18731 | 8.97MB | → PROMOTED | +0.00011 vs Leg 3 seed=444. Reference parent for BW5. |

---

## Thread: Compile / Fullgraph

Hypothesis: `torch.compile(fullgraph=True)` eliminates graph breaks → faster step → more steps in budget → lower BPB.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| 2026-03-29 | **BW5** (former champion) | BW4 + COMPILE_FULLGRAPH=1 | — | ✓ 74.52ms | **1.18672385** | **8.61MB** | → PROMOTED | −0.00058 vs BW4. 0 graph breaks. Roundtrip eval 2.77× faster. Seed=300 ⚠️ +0.00012 vs Leg 3 (mean still better). |

Notes: BW5 seed=300 does NOT individually confirm vs Leg 3. Mean is better (1.18715 vs 1.18743). A future leg should try to close this seed disparity.

---

## Thread: Cannon (gate feed-forward type)

Hypothesis: scalar cannon feed-forward gives the crawler loop a faster compiled path and calibrated output scale.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | Full Run BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|------------------------|------|---------|-------------|
| 2026-03-31 | BW5_Cannon | BW5 + CRAWLER_CANNON_TYPE=scalar | ✓ | ✓ speed | 1.18692423 | 8.44MB | ✗ DOES NOT PROMOTE | Gate signal (−0.00016) reversed at full run (+0.00020 vs BW5). Cross-run variance swamped cannon signal. No step overhead. Size actually −179KB vs BW5 at full run. |

8GPU gate detail (seed=444, 2000 steps):
- BWVC-00 control: 74.84ms · raw_bpb 1.28870981 · int6_sw_bpb 1.28787686
- BWVC-01 scalar cannon: 74.81ms · raw_bpb 1.28854887 · int6_sw_bpb 1.28820687
- Delta: −0.03ms speed (passes) · raw_bpb −0.00016 · size +343KB

Full run detail (seed=444, 600s, 8034 steps):
- BW5_Cannon: 74.69ms · raw_bpb 1.1990 · int6_sw_bpb **1.18692423** · 8,845,120 bytes
- BW5 champion: 74.68ms · raw_bpb 1.1987 · int6_sw_bpb **1.18672385** · 9,024,399 bytes
- Delta: +0.01ms · +0.00020 BPB · −179KB (zstd artifact smaller despite cannon)

---

## Thread: N-gram Embedding Enrichment

Hypothesis: richer n-gram hashing at the input (bigram → bigram+trigram) gives the crawler better local context at zero parameter cost.

| Date | Leg | Change vs Parent | 8GPU Gate | Full Run BPB | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-------------|------|---------|-------------|
| 2026-03-31 | BW6_Skipgram | BW5 + TRIGRAM=1 | ✗ null | — | — | ✗ DOES NOT PROMOTE | Null result. +0.0005 raw / +0.00014 int6_sw — both within variance noise. Speed: −0.06ms (zero overhead). Size: −140KB (interesting compression artifact). Crawler recurrence already approximates trigram context; static hash adds nothing. |

Gate detail (seed=444, 2000 steps, 8×H100):
- BW6SK-00 control: 74.53ms · raw_bpb 1.3083 · int6_sw_bpb 1.28951966 · 9,482,608 bytes
- BW6SK-01 trigram: 74.47ms · raw_bpb 1.3088 · int6_sw_bpb 1.28965847 · 9,342,986 bytes
- Delta: −0.06ms · +0.0005 raw · +0.00014 int6_sw · **−140KB**

---

## Thread: MLP Choke Architecture

Hypothesis: pyramid-shaped MLP bottleneck (CRAWLER_MLP_CHOKE_DIM=512) gives the loop
more representational pressure at each recurrence → better quality.

VERDICT: Current implementation (CHOKE_DIM=512) is **incompatible**. Concept not dead — see notes.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| 2026-03-31 | BW5_Pyramid | BW5 + CHOKE_DIM=512, shape=pyramid, groups=8 | ✓ −0.00987 int6_sw_bpb | SKIPPED | — | — | ✗ CONCEPT DEFERRED | Proxy inflation trap. 1GPU signal strong but 8GPU proxy run (via PyramidCannon control) shows cold param burden dominates at 2000 steps. |

1GPU gate detail (500 steps, seed=444, grad_accum=8):
- BWVP-00 (flat): step_avg 583.99ms · int6_sw_bpb 1.44668780
- BWVP-01 (pyramid): step_avg 611.21ms · int6_sw_bpb 1.43681894
- Delta: +27.22ms (÷8 ≈ +3.4ms real) · −0.00987 bpb (MISLEADING — see PyramidCannon)

Pyramid future paths (if revisited):
- Smaller choke dim (128 or 256) — less cold param burden
- Warm initialization of bottleneck weights
- Dedicated LR schedule for choke layers
- Investigate benefit at very long training (>>8000 steps)

---

## Thread: Combined — Pyramid + Cannon

Two-variable combined test. Cannon already validated for speed; pyramid sought quality synergy.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| 2026-03-31 | BW5_PyramidCannon | BW5 + CHOKE_DIM=512 + CANNON_TYPE=scalar | ✓ −0.0091 | ✗ +0.03440 int6_sw_bpb | — | — | ✗ DOES NOT PROMOTE | Hard failure. Proxy passed but 8GPU decisive regression. Root: 1.57M cold choke params compound over time. Crossover ~step 500, diverges through step 2000. |

8GPU gate detail (seed=444, 2000 steps):
- BWVPC-00 control: 74.40ms · raw_bpb 1.3069 · int6_sw_bpb 1.28787686 · 9,415,826 bytes
- BWVPC-01 pyramid+cannon: 79.33ms · raw_bpb 1.3283 · int6_sw_bpb 1.32227987 · 10,408,358 bytes
- Delta: +4.93ms · +0.0214 raw_bpb · **+0.03440 int6_sw_bpb** · +993KB

---

## Thread: BW7 MegaGate — 8-arm ablation (4×GPU SDPA, 2000 steps)

8 arms tested simultaneously on 4×GPU with SDPA fallback. Environment degraded vs standard
(no FA3, 4× not 8× GPUs) but cross-arm deltas are valid. Control confirmed comparable to
BW5 gate baseline. All zero-init mechanisms — zero param cost at step 0, benefit accrues.

| Date | Arm | Change vs BW5 | int6_sw_bpb | delta_vs_ctrl | Verdict |
|------|-----|--------------|-------------|----------------|---------|
| 2026-03-31 | CTRL-00 | baseline | 1.28912666 | — | control |
| 2026-03-31 | SMEAR-01 | CRAWLER_LOOP_SMEAR=1 | 1.28910136 | −0.00003 | ✗ NULL — CLOSED |
| 2026-03-31 | TAP-02 | TAP_DIM=32 per-loop | 1.28772880 | −0.00140 | ✓ signal |
| 2026-03-31 | **TAP-03** | **TAP_DIM=32 shared** | **1.28560427** | **−0.00352** | **→ BASELINE** |
| 2026-03-31 | TAP-04 | TAP_DIM=16 per-loop | 1.28646248 | −0.00266 | ✓ signal (weaker) |
| 2026-03-31 | ANC-05 | ANCHOR_DIM=32 | 1.28578393 | −0.00334 | ✓ strong signal |
| 2026-03-31 | ANC-06 | ANCHOR_DIM=64 | 1.28749998 | −0.00163 | ✓ moderate |
| 2026-03-31 | FLAT-07 | FLAT_WEIGHT_SHARE=1 | 1.32606772 | +0.03694 | ✗ HARD FAIL — DEAD |

Key findings:
- **TAP shared >> TAP per-loop.** One shared encoder anchor beats 3 specializing at dim=32.
- **Anchor dim=32 >> dim=64.** Loop-to-loop causal write state is low-dimensional.
- **SMEAR null.** Loops already intentionally distinct via FLOW. Mechanism had nothing to correct.
- **SharedFlat catastrophic.** −4.7M params destroyed performance. Flat layers do genuinely distinct work.
- **TAP-03 baked into new baseline (BW8).** Not yet a full run — stacking signals first.
- **ANC-05 next.** Test anchor on top of BW8 (TAP) baseline as BW9.

---

## Thread: Encoder Tap

Hypothesis: per-loop gated access to frozen intermediate encoder representations gives the
crawler a stable, pre-quantization anchor to check against as it loops.

| Date | Leg | Change vs Parent | Gate | BPB | Size | Verdict |
|------|-----|-----------------|------|-----|------|---------|
| 2026-03-31 | BW7 MegaGate TAP-03 | BW5 + TAP_DIM=32 shared | ✓ −0.00352 (4GPU proxy) | — | 9.5MB | → PROMOTED TO BASELINE |
| — | BW8_Tap | BW5 + TAP_DIM=32 TAP_LOOP_SPECIFIC=0 | pending 8GPU | pending | — | ⏳ SIGNAL STACKING |

Decision: TAP-03 signal strong enough (12× noise floor) to bake into baseline and hunt further
signals on top before committing to a full production run.

---

## Thread: Delta Anchor

Hypothesis: per-loop causal write state — each loop commits a small anchor for the next loop
to condition on. Battery differentiates reading; anchor differentiates writing.

| Date | Leg | Change vs Parent | Gate | BPB | Size | Verdict |
|------|-----|-----------------|------|-----|------|---------|
| 2026-03-31 | BW7 MegaGate ANC-05 | BW5 + ANCHOR_DIM=32 | ✓ −0.00334 (4GPU proxy) | — | 9.4MB | ⏳ NEXT GATE on BW8 base |

Next: BW9_Anchor — test ANCHOR_DIM=32 on top of BW8 (TAP) baseline.

---

## Planned Hypotheses

| Priority | Hypothesis | Thread | Prerequisite | Rationale |
|----------|-----------|--------|-------------|-----------|
| 1 | BW9 — Anchor on BW8 | Recurrence | BW8 (TAP baked in) | ANC-05 showed −0.00334. Test on new TAP baseline. Stack delta → justify full run. |
| 2 | BW10 full run | Production | BW9 gate pass | If TAP+Anchor stack, combined delta justifies 8×GPU 600s run. |
| 3 | Coprime loader port | Data | BW8 | Neural SOTA has coprime, crawler uses sequential. Worth gating. |
| 4 | Warmdown tuning | Schedule | BW8 | BW5 seed=300 ⚠️. Carry forward to BW8. |

---

## Thread: Trapper Keeper 1 — Multi-Crawler (8F+3C)

Hypothesis: Optimal crawler config is NUM_FLAT_LAYERS=8, NUM_CRAWLER_LAYERS=3 (C=loops symmetry).
Based on 20-arm layer relationship grid mapping flat(5-9) × crawler(1-4) surface.

| Date | Leg | Change vs BWX 9F | Environment | int6_sw_bpb | Size | Step_ms | Verdict |
|------|-----|-------------------|-------------|-------------|------|---------|---------|
| 2026-04-09 | Grid 8F+3C | FLAT=8, CRAWL=3 | 2xGPU, 1000 steps | 1.39529 | 14.58MB | 181 | grid best |
| 2026-04-09 | Isolated 8F+3C | FLAT=8, CRAWL=3 | 4xGPU, 1000 steps | 1.34632 | 15.00MB | 574 | confirmed |
| 2026-04-09 | **TK1 production** | **FLAT=8, CRAWL=3** | **8xH100, 600s, no FA3** | **1.13527** | **17.95MB** | **157** | **QUALITY PASS, SIZE FAIL** |
| 2026-04-10 | **TK1 recovered legal** | **FLAT=7, CRAWL=3, GPTQ+pyminify+prune** | **8xH100, 600s, FA3** | **1.13541** | **15.90MB** | **149** | **QUALITY PASS, LEGAL, confirmed on 300 and 4** |

Production quality beats BWX 9F by -0.00341. Artifact over 16MB with zstd. Brotli recompress pending.
Pod lacked FA3 — with FA3, step time would be ~110ms → ~5,450 steps (vs 3,811). Quality would improve.

---

## Thread: Layer Relationship Grid

5×4 grid: NUM_FLAT_LAYERS(5-9) × NUM_CRAWLER_LAYERS(1-4), 2xGPU, 1000 steps.

Best int6_sw_bpb per row (delta vs 9F+1C control at 1.41256):

| Config | int6_sw | delta | step_ms | size_MB |
|--------|---------|-------|---------|---------|
| 8F+3C | 1.39529 | -0.01727 | 181 | 14.58 |
| 7F+4C | 1.39547 | -0.01709 | 206 | 15.01 |
| 9F+3C | 1.39599 | -0.01657 | 198 | 15.33 |
| 7F+3C | 1.39910 | -0.01346 | 176 | 13.79 |
| 6F+4C | 1.40000 | -0.01256 | 207 | 13.90 |

Key findings:
- Quality ridge: 8F+3C → 7F+4C → 7F+3C → 6F+4C
- 3C is optimal at 3 loops (C=loops symmetry). 4C reverses on 9F row.
- Below 7F, quality drops off. Crawler can't compensate for too few flat layers.
- Trading flat for crawler: free at 3C (8F→7F costs nothing), expensive at 1C.

---

## Thread: Corpus Ablations v1

16-arm screen on BWX 9F base (4xGPU, 1500 steps).

| Arm | Change | int6_sw delta | Verdict |
|-----|--------|---------------|---------|
| A07 2 crawler layers | NUM_CRAWL=2 | -0.0119 | BREAKTHROUGH |
| A04 4 loops diff | LOOPS=4, ROPE=9,3,1,1 | -0.0046 | strong |
| A03 4 loops naive | LOOPS=4, ROPE=9,1,1,1 | -0.0040 | strong |
| A05 5 loops prog | LOOPS=5, ROPE=9,5,3,1,1 | -0.0037 | good but < A04 |
| A02 Anchor | ANCHOR_DIM=32 | -0.0024 | positive |
| A10 QAT softclamp | QAT_SURROGATE=softclamp | -0.0047 vs legacy | strong |
| A01 TAP shared | TAP_DIM=32 | -0.0010 | mild |
| A08 Crawler int8 | CRAWLER_QUANT_INT8=1 | -0.0006 | neutral (size tool) |
| A06 INST_DIM=64 | INST_DIM=64 | +0.0011 | DEAD |
| A11 QAT sigmoidste | QAT_SURROGATE=sigmoidste | +0.041 | DEAD (unstable) |

---

## Thread: Symmetry (C=LOOPS)

Tested 4×4, 6×6, 8×8 on 8F base. Only 4×4 completed (806ms/step, impractical).
3×3 confirmed optimal under wallclock pressure. Higher symmetry orders too slow.

---

## All-Time Reference

| Leg | BPB (seed 444) | Size | Mean BPB | Status |
|-----|----------------|------|----------|--------|
| **TK1 7F+3C + GPTQ + pyminify + prune** | **1.13541288** | **15.90MB** | **1.13643599** | **Current in-tree leader** |
| BWX 9F | 1.13867894 | 15.24MB | pending (seed 300) | Former in-tree leader |
| TK1 (8F+3C) | 1.13526829 | 17.95MB (OVER) | — | quality pass, size fail — brotli pending |
| Leg 3 | 1.18720 | 8.84MB | 1.18743 (3-seed) | Former champion |
| BW4 | 1.18731 | 8.97MB | — | Superseded |
| BW5 | 1.18672 | 8.61MB | 1.18715 | Former champion |
| BW8_Tap | pending full run | ~9.5MB est | — | working baseline (TAP baked in) |
| BW22 A3 gate arm | 1.24091238 (2k gate) | gate-only | — | promoted quality candidate for 4h path |

<!-- NEW LEG STUB -->
| 2026-04-09 | BW_9F2C | (fill in) | ⏳ | ⏳ | — | — | ⏳ PENDING | |
