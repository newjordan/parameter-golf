# Tomorrow Morning Handoff — 2026-04-17 evening → 2026-04-18 AM

## TL;DR
- Current champion: **val_bpb 1.89896** (tag `champ_qk05`, QK_GAIN_INIT=0.5, L=8 MLP=2 CHAOS=5 MATRIX_LR=0.035).
- Artifact: 13.75 MB int8+zlib (2.25 MB headroom).
- v17 completed; pod paused.
- New analysis, scripts, and design docs ready to fire. Fire order: **v18 → v19 → v21 (after code change) / v20 (after tokenizer build) → kernel work (D1 input reinjection)**.

## What landed today
- **ChatGPT deep research** saved at `research_brief/external_responses/chatgpt_deep_research_01.md`. Validates v16 SIREN disaster (β>>1 is wrong) and recommends conservative init α∈U(-0.10,0.10), β∈U(0.8,1.2), φ∈U(-π,π).
- **CHAOS_SCALAR_MODE gate** added at `test_vortex_2k.py:636`. Set env `CHAOS_SCALAR_MODE=conservative` to use the new init; `CHAOS_SCALAR_MODE=randn` for legacy; omit for legacy default.
- **Ablation queue** drafted at `research_brief/ablation_queue.md` (full Q1-Q10 breakdown, tiered HIGH/MED/LOW/KERNEL).
- **Kernel design docs**:
  - `research_brief/Q1_stability_design.md` (spectral norm on W slice — from earlier subagent work).
  - `research_brief/Q3_input_reinjection_design.md` (NEW — input reinjection kernel change, the highest-EV theoretical win).
  - `research_brief/Q4_quant_diagnostic.md` (earlier — all-int8 per-channel is correct path for vortex).
- **Ready-to-fire scripts**:
  - `run_v18_conservative_siren.sh` — 8 trials, conservative chaos-scalar init + SCALAR_LR + deeper CHAOS probes.
  - `run_v19_qk_fine_sweep.sh` — 6 trials, QK_GAIN fine sweep around 0.5.
  - `run_v20_vocab4k.sh` — 6 trials, 4k vocab + depth/MLP reinvestment. **BLOCKED: needs 4k tokenizer + shards (only 1k + 8k exist on pod).**
  - `run_v21_muon_wd.sh` — 5 trials, Muon weight decay. **BLOCKED: Muon class doesn't accept weight_decay; needs code change in test_vortex_2k.py:114.**

## Fire order tomorrow

### First hour
1. **Unpause pod** (vast-dealer 4x).
2. **Start v18** (conservative SIREN at champ): `cd /workspace/sota_crawler && ./run_v18_conservative_siren.sh > logs/v18_driver.log 2>&1 &`. ~15 min for 8 trials × 4 GPUs.
3. **While v18 runs**, inspect `kernels/vortex_fused.py:153-168` and `research_brief/Q3_input_reinjection_design.md`. Decide whether to task a subagent with the input-reinjection kernel change. This is the highest-EV lever but requires 2 hours + param budget reallocation.

### Second hour
4. Parse v18 results. If conservative mode beats randn by ≥0.003 BPB on any of the 3 seeds, adopt it.
5. **Start v19** (QK fine sweep). 6 trials × ~7 min ≈ 10 min wallclock.
6. Optional: build 4k SentencePiece tokenizer + retokenize shards in background (unblocks v20).

### Third hour
7. If v18 + v19 both moved the needle, combine winners into a new champion and log.
8. Attempt v21 (Muon WD) only after adding weight_decay to Muon class. Subagent can do this safely: decoupled `p.mul_(1 - lr*wd)` before the update in `step()`.
9. Pursue D1 (input reinjection) in isolation if the first-hour subagent decision was "go."

### Parallel (any time)
- D4 (Nsight Compute roofline): one-off profiling run. Informs D5 (Triton autotune grid).
- D5: Triton autotune in kernels/vortex_fused.py. Good follow-on once a kernel subagent is paired.

## Pod state
- Host: `vast-dealer` (4×H100 80GB).
- Current branch: `TON-E`.
- Workdir: `/workspace/sota_crawler`.
- v17 results written to `/workspace/sota_crawler/logs/v17_suite/`. Chain log at `/workspace/sota_crawler/logs/chain/`.
- Home repo: `/home/frosty40/sota_crawler` on `TON-E`. Scripts/docs drafted here; **need rsync to pod before firing**.

## Known-closed paths (do not retest)
- Whale hybrid (1.4229 post-quant disaster).
- Ouroboros / Helix / Smokestack.
- `CHAOS_SIREN_OMEGA0` ≥ 10 (β>>1 catastrophic).
- SCALAR_LR ∈ {0.01, 0.02, 0.08, 0.16} at old champion.
- BETA2 = 0.97 at old champion (v15).
- VORTEX_BLOCK_SIZE = 256 (OOM).
- VORTEX_FWD_NUM_STAGES = 4 at champion (v17 shows 2389 ms/step vs baseline 1480 ms/step — memory-pressure regression).

## What needs a human decision before tomorrow
- Pursue D1 (input reinjection)? **Recommend yes.** Estimated 2 hours subagent time + 2.1 MB int8 budget cost (would force L=7 or d_model=448). Pairs with the strongest theoretical prediction in the whole research round.
- Build 4k tokenizer tonight, or morning? **Recommend morning** — it's a 15-25 min precheck, better to start fresh and let v18 prove the chaos-scalar premise first.

## Final v17 result (spot check)
- `fwp4` (FWD_NUM_WARPS=4): val_bpb 1.9009 → int8 roundtrip 1.9013, 354k tok/sec. Confirms current vortex quant pipeline only loses +0.0004 BPB to int8+zlib — the Q4 diagnostic is resolved for vortex (naive int6 is NOT the default path for vortex; that was whale-specific).
