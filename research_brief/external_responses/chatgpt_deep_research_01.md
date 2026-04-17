# ChatGPT Deep Research Response 01

**Source:** `https://chatgpt.com/s/t_69e1d029a8e48191802985452577a69b` (shared 2026-04-17)
**Pulled into repo:** 2026-04-17
**Validation note:** this is the response that *agrees* with our v16 SIREN disaster. v16 data at time of saving: baseline_randn=1.9016, siren_o15=3.11, siren_o20=2.45, siren_o25=1.97 — i.e. the larger β ranges recommended by the first wave of research agents destroyed the model. This deep-research response explicitly warns against that and recommends β ∈ U(0.8, 1.2). It was right.

---

## Project frame
The brief hosted on GitHub sets an unusually strict objective: beat the current 1.9012 val-BPB champion within a roughly 10-minute train-plus-serialize budget on the OpenAI challenge rig of 8×H100, while keeping the final compressed artifact under 16 MB. The present winner already lands at about 13.75 MB, which means the project is not primarily size-starved today; it is **BPB-starved**, and only secondarily artifact-starved. That changes how aggressive you should be with quantization and tokenizer compression.

The highest-value reading of the repo evidence is that the current chaos loop is still underidentified. The brief itself says the largest unswept axes are SCALAR_LR, chaos-scalar initialization, quantization, and the possibility that the chaos primitive is the wrong thing to spend compute on at all. It also says the strongest existing negative result is that explicit layer count beat raw CHAOS_DEPTH expansion in the "one fat chaotic block" regime. That should push the next round of work toward **mechanism-fixing experiments**, not blind depth scaling.

## Recurrence and depth

### Q1 — tied-iteration recurrence stability

**Findings.** Your update rule
```
b_{i+1} = tanh(W b_i) + α sin(β b_i + φ)
```
is DEQ-like in the one thing that matters most: it is a tied map repeatedly applied until useful computation emerges, or fails to. In the DEQ literature, forward usefulness and backward stability both depend on the fixed-point map being contractive enough near the operating regime, which is why later DEQ work adds explicit Jacobian regularization, and why the reference DEQ code tracks the relative residual `|f(z)-z|/|z|` as the basic convergence signal. A clean local heuristic for your cell is that the Jacobian norm is bounded by roughly `||W||₂ + |αβ|`: the tanh branch contributes at most `||W||₂`, while the sine branch contributes at most `|αβ|` because the derivative of `sin(βb+φ)` is `β cos(βb+φ)`. That bound is my inference from the cell definition, not a theorem stated in the papers, but it is the right lens for deciding whether the loop is convergent, marginal, or chaotic.

The repo's own note that about 30% of seeds land near the "chaotic-edge regime" is exactly what I would expect from a recurrence whose scalar term is initialized by an unconstrained Gaussian draw. In DEQs, that kind of variance is survivable when the solver and Jacobian regularization are doing real work; in your megakernel, where the chaos loop is just unrolled a small fixed number of times, it is more likely to manifest as either fast collapse to a trivial attractor or unstable oscillation that never becomes useful depth. That makes Q1 and Q2 tightly coupled: the problem is not only optimization, it is the operating point of the recurrence itself.

**RECOMMEND.**

- Add a cheap convergence diagnostic first, before a bigger ablation: in `kernels/vortex_fused.py` at the chaos-loop site the README points to, export per-iteration means of `||b_{i+1}-b_i|| / (||b_i|| + ε)` and `cos(b_{i+1}, b_i)` for one debug batch. If residuals drop rapidly to near-zero by step 2 or 3, extra CHAOS_DEPTH is mostly dead compute; if they oscillate or grow, you have a stability problem, not an expressivity problem. I would treat this as a must-have diagnostic, not an optional nicety.
- Add a contractivity regularizer in `test_vortex_2k.py`: one power-iteration estimate of `||W||₂`, plus a scalar penalty on `max(0, ||W||₂ + |αβ| - 0.95)^2`. My expectation is modest at current depth 5, but meaningful as an enabler for depth 7–15: about **0 to 0.003 BPB** at fixed depth 5, and **0.005 to 0.015 BPB** if it unlocks deeper useful recurrence.
- If the diagnostic shows immediate collapse, treat that as a negative result on "raw extra depth" and stop spending trials on CHAOS_DEPTH>10 until the cell is redesigned.

**Priority.** High. This is the fastest way to distinguish "recurrence is good but unstable" from "recurrence is currently inert."

### Q2 — SIREN-style initialization for the sine term

**Findings.** The SIREN result is not that "sine is magic"; it is that periodic nonlinearities are unusually sensitive to initialization scale, because the wrong frequency/amplitude regime either kills gradients or throws the network into pathological oscillation. That applies even more strongly in your case, because the sine term is not a plain feed-forward activation but part of a recurrent update loop. In your cell, β is the nearest analogue of SIREN's frequency control, α is the amplitude knob, and φ is a phase offset. **What absolutely does not transfer from SIREN is the large first-layer ω₀ intuition as a literal number:** your sine is inside a tied recurrent map, so large initial β or large initial αβ is exactly what pushes you toward the "chaotic-edge" seed failures the repo already reports.

The most useful SIREN lesson here is therefore qualitative, not literal: start the periodic branch in a small-amplitude, controlled-frequency regime where it perturbs the tanh iteration without dominating it. The current `torch.randn(3)` almost certainly makes α too large too often. Because your loop also has the bounded-output property the brief highlights, I would optimize for "stable low-amplitude perturbation that can grow if useful," not "maximal frequency richness at step 0."

**RECOMMEND.**

- Replace the current scalar init in `test_vortex_2k.py` with
  ```
  alpha ~ U(-0.10, 0.10)
  beta  ~ U(0.8, 1.2)
  phi   ~ U(-π, π)
  ```
  and keep the sine branch centered. This is a direct drop-in replacement for the current Gaussian init. I would budget this as a **0.003 to 0.010 BPB** seed-stability play, mostly by cutting bad initializations rather than by raising the model's asymptotic ceiling.
- Pair that init with either unchanged BETA2=0.95 or a separate chaos-scalar optimizer group rather than an immediate global BETA2 increase. A quieter init plus a much slower second-moment estimator is the wrong combination.
- Do not import a literal SIREN ω₀=30 mindset into this recurrent cell. In this architecture, that is far more likely to destabilize the recurrence than to help it.

**Priority.** High. This is cheap to implement, directly addresses a repo-flagged failure mode, and interacts favorably with Q1.

### Q3 — recurrent depth precedent and whether the current loop is "doing recurrence right"

**Findings.** The strongest modern recurrent-depth precedent is not just "tie a block and run it many times." The 2025 recurrent-depth work trains the loop with three design choices that look foundational, not cosmetic: it **re-injects the input/latent embedding at every recurrent step**, it **samples recurrence counts during training** rather than using one fixed unroll length, and it **truncates backprop through only the last few iterations** to keep memory stable. Just as important, that work explicitly reports that feeding the current step index into the core interacted badly with path independence and hurt extrapolation. In other words, the success recipe is "input-conditioned iterative refinement with depth stochasticity," not "autonomous state evolution inside a tied loop."

That matters because your current chaos loop, as described in the brief, is much closer to an autonomous attractor with structured perturbation than to the recurrent-depth recipe that scaled. That is the most plausible explanation for your own empirical contradiction: NUM_LAYERS beats CHAOS_DEPTH because added explicit blocks keep interacting with the token stream, whereas additional chaos iterations mostly refine the same local state without new input-conditioned computation. I would therefore not conclude that recurrent depth is dead at your scale. I would conclude that the current cell is missing the most important mechanism the successful recurrent-depth papers use.

**RECOMMEND.**

- Change the recurrence from "autonomous" to input-conditioned inside `kernels/vortex_fused.py`:
  ```
  b_{i+1} = tanh(W b_i + U x_layer) + α sin(β b_i + φ)
  ```
  where `x_layer` is the same layer input injected at every chaos iteration. This is the single most important architectural copy from the recurrent-depth literature. My estimate is **0.005 to 0.020 BPB** upside if recurrence is salvageable at all.
- Add random depth training for the chaos loop rather than only fixed CHAOS_DEPTH=5: a small distribution over `{3,5,7,9}` is the pragmatic 10-minute-budget version of the recurrent-depth recipe. Couple it to truncated BPTT through only the last 2–3 iterations.
- If you do not add per-step input reinjection, then I would lower the priority of further CHAOS_DEPTH sweeps. In that case, the repo's own evidence already says explicit depth is the better effective-depth axis.

**Priority.** High. This is the best "mechanism repair" move in the whole brief.

## Optimization and averaging

### Q5 — why BETA2=0.95 is low

**Findings.** In Adam, β₂ is the EMA decay on squared gradients, so it determines how quickly the denominator reacts to changing gradient scale. The original Adam paper explicitly motivates Adam as a method for noisy and non-stationary objectives, which is exactly the framing under which a low β₂ makes sense. In your schedule, the total sweep is only 300 steps with 20 warmup steps and a strong cooldown; mathematically, β₂=0.95 has an effective averaging window of about 20 steps, 0.97 about 33, 0.99 about 100, and 0.999 about 1000. On a 300-step run, 0.999 is almost certainly too inert for the scalar/head/embed side, especially if Muon is rapidly moving the matrix parameters underneath it.

There is also a code-path reason not to panic about the low number. The brief says Muon handles matrix parameters while Adam handles embeddings, head, and scalars; the most non-stationary pieces of your model are exactly the scalar chaos parameters and tied embedding/head-adjacent tensors, so they are the ones most likely to prefer a shorter second-moment memory. I would read 0.95 not as "weird," but as a plausible short-horizon LM choice that is worth nudging toward 0.97, not toward 0.99.

**RECOMMEND.**

- Keep the global sweep centered on BETA2 in `{0.95, 0.97}`, and demote 0.99 to a lower-priority boundary check. My expectation is that 0.97 is the only realistic challenger; I would treat 0.99 as more likely to help only if current scalar LRs are too aggressive. Expected delta: **0 to 0.004 BPB** either way.
- Better than a global move: create a separate Adam group for the chaos scalars and keep them at β₂=0.95 while trying β₂=0.97 for the rest of the Adam side. That is more consistent with the hypothesis that the sine-branch parameters are the only truly fast-changing scalar state.
- I would not move the whole model toward a "standard" β₂≈0.999 recipe unless the run horizon changes dramatically.

**Priority.** Medium-high. Worth tuning, but I would spend fewer trials here than on Q1–Q3.

### Q6 — why EMA probably hurt on whale

**Findings.** The strongest recent EMA result is not that EMA always helps, but that it helps with the right decay and horizon. The 2024 EMA study shows that EMA naturally reduces stochastic noise and can substitute for some learning-rate decay, but it also shows a very important practical tradeoff: slower EMA decays peak later. On a short run, that means a too-slow EMA is stale and overweights bad early iterates. Your repo evidence is consistent with exactly that failure mode: whale had a short overall horizon relative to common EMA settings, and the brief reports EMA hurt by +0.0168 BPB instead of helping. That is much more suggestive of a horizon mismatch than of "EMA is fundamentally bad for this model family."

Because your scoring condition is train-fast-then-serialize-fast, the right EMA, if any, is not the long-horizon teacher EMA common in long pretraining runs. It is a late-start short-horizon EMA used only for the last checkpoint evaluation and serialization. I would also keep EMA away from any weird interaction with quantization until the quant path is made reliable; the whale result is too confounded to say more than that.

**RECOMMEND.**

- For 300-step vortex sweeps, if you test EMA at all, use a late-start EMA: start around step 180 and use decay 0.99 to 0.995, not 0.999+. For 1500-step whale-style runs, shift to 0.995 to 0.998 with start around 60% of training.
- Apply EMA only for final validation and serialization, not as an always-on training companion.
- Until quantization is fixed, my default recommendation is effectively "EMA off by default." I would only re-enable it after Q4 is under control.

**Priority.** Medium. Fix later than recurrence and quantization.

### Q7 — Muon at small scale and tied embeddings

**Findings.** The main concern named in the brief — that tied embeddings might violate Muon's assumptions — appears to be a red herring for the current vortex path, because the brief says Muon is used for matrix parameters while Adam is used for embeddings, head, and scalars. That means the tied input/output embedding is not being orthogonalized by Muon in the first place. So the "tied-embed gradient sum before Muon orthogonalization" failure mode is not the current code path.

The more relevant Muon question is whether it is worth replacing at this scale. I do not see strong evidence for that. The public Muon writeup and the "Muon is Scalable for LLM Training" paper both say the optimizer is specialized for 2D hidden-layer matrices, and the scaling paper's main work was extending it upward in scale by adding weight decay and correcting update scaling, not showing that it breaks at small scale. Combined with the repo's own result that Muon is part of the current best setup, the burden of proof has flipped: optimizer swaps are now low-priority unless you have evidence of a specific pathology.

**RECOMMEND.**

- Keep Muon on block matrices and Adam on embeddings/head/scalars. I would not spend many trials on Adam-only or Shampoo-style replacements until Q1–Q4 are resolved.
- If you want one Muon-side optimization test, make it **weight decay**, not optimizer replacement. The scaling paper identifies weight decay as one of the two crucial ingredients that let Muon scale cleanly. I would expect **0.002 to 0.008 BPB** upside from a good Muon-WD sweep before I would expect anything from a full optimizer swap.
- Do not move tied embeddings into Muon. That would create a new risk surface without evidence of upside.

**Priority.** Medium-low. Muon looks more like a stable base than a current bottleneck.

## Quantization and vocabulary

### Q4 — quantization failure mode at int6

**Findings.** The official GPTQ code path is a very strong clue. Its default layer discovery function only looks for `nn.Conv2d` and `nn.Linear`. If a model family packages its effective projections inside unsupported custom blocks or fused modules, GPTQ calibration can trivially find zero layers. That matches the whale log the brief quotes almost too perfectly: "calibrated 0 layers" followed by fallback to catastrophic naive low-bit quantization. So the most likely root cause is not that GPTQ is intrinsically bad, but that the model family being quantized did not expose its weights in the module pattern GPTQ expects.

The second clue is the size budget. Your current vortex winner is already under the 16 MB cap with about 2.25 MB of headroom. That means a precision drop from int8 to naive int6 only makes sense if it buys something structurally important, such as a much larger model or a much larger vocabulary. Otherwise it is exactly the wrong trade: low bits amplify outlier-channel error, and the low-bit quantization literature says preserving a few salient channels is often the whole game. SmoothQuant identifies activation outliers as a central problem for INT8 activation quantization, AWQ shows that protecting only a tiny fraction of salient weights can dramatically reduce quantization error at low bit-width, and LLM.int8 explicitly isolates outlier dimensions instead of quantizing them naively. Against that background, a "naive int6 on 4 tensors" fallback is almost guaranteed to be brittle.

**RECOMMEND.**

- For vortex, stop treating int6 as the default target. Use **all-int8, per-row or per-channel scales** for every 2D tensor, and keep tiny/control tensors in higher precision if needed. Given the present artifact headroom, this is the highest expected-value recipe. I would model this as **near-zero to +0.01 BPB** quantization loss if implemented well, versus the repo's already-observed catastrophic downside when low-bit fallback goes wrong.
- For whale or any future GPTQ/AWQ path, either expose quantizable projections as real `nn.Linear`-like modules during calibration or write a custom layer finder. Otherwise, do not trust "GPTQ succeeded" unless the calibrated-layer count is nonzero.
- For compression, use brotli only for final submission artifacts if the wall-clock overhead stays inside budget; keep zlib for sweep-time iteration.

**Priority.** High. This is the cleanest immediate risk reduction in the entire project.

### Q8 — tokenizer choice and embedding budget

**Findings.** Even from the repo brief alone, the embedding budget matters a lot. The model uses 8 heads with head-dim 64, so the hidden size is 512; with tied embeddings and an 8192-token vocabulary, that is a single 8192×512 embedding matrix, or about 4.19M parameters before any other block weights are counted. That is exactly the kind of regime where vocabulary and model-capacity allocation become entangled. The NeurIPS vocabulary-scaling paper's central result is that larger models deserve larger vocabularies; the contrapositive is the helpful one for you: **small models deserve smaller vocabularies** than frontier recipes use. At the same time, byte-level models remove vocabulary parameters but make sequences much longer, and byte-level papers consistently flag sequence-length cost as the main drawback unless you also redesign the architecture around downsampling or pooling.

That pushes the near-term answer away from both extremes. I would not go to 16k on this tiny, time-capped regime, because that spends precious artifact and parameter budget where the scaling-law evidence says bigger vocabularies help more at bigger model scales. I also would not jump to a pure byte-level model inside the current architecture, because the sequence-length tax is directly at odds with a 10-minute wall-clock objective. The most reasonable search region is therefore **4k–8k BPE/SentencePiece**, with **4k as the high-value challenger**.

**RECOMMEND.**

- Run `VOCAB_SIZE=4096` as the primary tokenizer ablation, not 16384 and not byte-level. A shift from 8192 to 4096 frees about **2.1M parameters / raw int8 bytes** in the tied embedding matrix, which can be reinvested into explicit layers or width. I would not expect a pure 4k swap by itself to be a slam-dunk, but **4k plus reinvestment in depth** is one of the most promising structural bets in the brief.
- De-prioritize byte-level tokenization unless you are also willing to adopt a pooling/downsampling architecture closer to the "super tiny language models" or tokenizer-free encoder work.
- Treat 16k vocab as low priority. Based on the scaling-law result, it is more likely to be correct after the model gets bigger, not before.

**Priority.** High. This is one of the few levers that can simultaneously improve BPB ceiling and make quantization easier.

## Kernel ceiling and light regularization

### Q9 — H100 throughput ceiling for the fused megakernel

**Findings.** The right way to think about the kernel ceiling is with a roofline-first Nsight workflow, not with raw tokens/sec alone. The Hopper tuning guide tells you the practical boundaries that matter most on H100: 64 warps per SM, a 64K-register file per SM, and up to 228 KB of shared memory per SM, with 227 KB addressable by one block. That makes your likely failure modes very standard: too many registers per thread, too much shared memory per block, or a fused kernel whose control flow and local state kill occupancy before arithmetic intensity can pay off. Nsight Compute's profiling guide and report views are specifically designed for this kind of analysis: roofline, source counters, scheduler stats, and function-level stall breakdowns.

The second part is Hopper-specific. Hopper adds TMA, and the official tuning guide explicitly says TMA enables warp-specialized code by offloading data movement so other warps can work on local data. The Triton tutorials reinforce the same set of ideas: persistent kernels, TMA/tensor-descriptor paths on Hopper, autotuning over num_warps and num_stages, program reordering for L2 locality, and explicit register-pressure management inside fused kernels. That makes the practical optimization stack fairly clear: first measure whether you are register-bound, smem-bound, or memory-bandwidth-bound; then test the Triton-side fixes that correspond to that bottleneck.

**RECOMMEND.**

- Profile one representative training step with Nsight Compute roofline + memory + scheduler sections and write down, for the chaos kernel only: achieved occupancy, registers/thread, smem/block, tensor utilization, DRAM throughput, L2 hit rate, and top stall reasons. Until you have those, "fraction of peak" is a guess.
- Add a Triton autotune grid around the fused kernel's tile shape, num_warps, and num_stages, if that is not already exhaustive. Typical upside for a not-yet-exhaustive fused kernel is materially positive. I would rank this as a plausible **5–15% throughput** bet.
- If the kernel is memory-bound, try a persistent/TMA-capable rewrite path and improve program ordering for L2 locality; if it is register-bound, trim live values in the chaos loop and move noncritical updates later in the loop. Combined upside from these changes is plausibly **8–20% throughput**, which on this project effectively means "free model size."

**Priority.** High. Every free percent of throughput buys a larger model under the fixed time cap.

### Q10 — first-layer noise and regularization in the short-run regime

**Findings.** The recent single-epoch LM pretraining evidence is very clear on one point: **dropout is not the regularizer you want here**. The 2025 "Drop Dropout" paper finds that for single-epoch pretraining, downstream LM quality and several downstream tasks improve when dropout is removed, and even "early dropout" underperforms no dropout. That is unusually aligned with your setup, because your sweep regime is short-horizon and explicitly not a long-convergence training job. Label smoothing is also a poor fit for a BPB-scored language model project: in LM settings it often worsens perplexity even when it improves calibration or downstream metrics.

Input noise is the only regularization idea in this bucket that still looks live, but the evidence is weaker and more indirect. NEFTune shows noisy embeddings can help instruction tuning, which at least supports the idea that tiny embedding perturbations can be useful; it does not prove the same thing for short-horizon pretraining. So I would treat first-layer noise as a small, carefully scaled perturbation test, not as a core optimization pillar. If SIGMA=0.01 is being applied directly in embedding space, that may already be too large for a first pass in a tiny model.

**RECOMMEND.**

- Keep dropout off and do not add stochastic depth as the next regularization sweep. The best current evidence points the other way for single-epoch LM pretraining.
- If you test embedding noise, start with `FIRST_LAYER_NOISE=1` and `SIGMA=0.003`, then bracket with `{0.001, 0.005}` before spending more time at 0.01. I would price this as a **0 to 0.004 BPB** opportunity, with meaningful downside if the sigma is too large.
- Do not spend cycles on label smoothing if the objective is validation BPB. Its most reliable wins are in calibration and some downstream settings, not in minimizing LM cross-entropy.

**Priority.** Medium-low.

## Ranked experiment queue

The strongest near-term queue, in descending order of expected value, is this.

1. **Repair the recurrence** before spending more on CHAOS_DEPTH. Add per-step input reinjection to the chaos loop, randomize depth during training over a small set such as `{3,5,7,9}`, and add a convergence diagnostic. This is the most evidence-backed way to make tied recurrence resemble the successful recurrent-depth literature instead of an autonomous attractor. **Expected upside: 0.005 to 0.020 BPB** if recurrence is salvageable.
2. **Replace `torch.randn(3)` chaos-scalar init**. Use a small-amplitude sine branch: `alpha∼U(-0.10,0.10), beta∼U(0.8,1.2), phi∼U(-π,π)`. **Expected upside: 0.003 to 0.010 BPB**, mostly by reducing bad-seed variance.
3. **Abandon naive int6 as the default deployment target.** For vortex, use all-int8 with per-row/per-channel scales and only revisit lower bits if a larger model forces it. **Expected value: mostly risk removal**, but possibly the single most important deployment fix.
4. **Run the 4k vocabulary structural ablation.** `VOCAB_SIZE=4096`, then reinvest the saved embedding budget into explicit depth rather than into deeper chaos. **Expected upside with reinvestment: 0.005 to 0.020 BPB**; without reinvestment, nearer flat.
5. **Do a real H100 roofline pass.** Measure whether the fused kernel is occupancy-, register-, smem-, or bandwidth-limited, then test Triton autotune plus persistent/TMA-style variants accordingly. **Expected upside: 5–20% throughput**, which should be translated directly into model-capacity experiments.
6. **Keep BETA2 low and EMA late or off.** Prioritize 0.95 versus 0.97, not 0.99, and if EMA is tested, make it a late-start short-horizon EMA.

**Bottom-line judgment:** the next real breakthrough is more likely to come from making the recurrence input-conditioned and stable, or from reallocating embedding budget into explicit model capacity, than from fine-tuning generic optimizer knobs.
