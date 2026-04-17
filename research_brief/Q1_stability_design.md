# Q1 — Chaos-loop stability regularizer (design)

## TL;DR — pick (a) spectral-norm on `W`, reject (b), defer (c)

The external agent's literal `torch.autograd.functional.jacobian(chaos_step, b)` cannot be computed: the chaos step exists only inside `_vortex_helix_fwd_kernel` at `kernels/vortex_fused.py:153-168`. Per-iterate `b_i` only lives in Triton registers; `torch.autograd` sees a single opaque `VortexHelixFunction.apply` (see `kernels/vortex_function.py:20-35`). Any attempt to hand a Python `chaos_step` closure to `torch.autograd.functional.jacobian` would either (i) double-compute the whole fwd in PyTorch eager at O(T·D²·depth) per step, blowing the 10-min cap, or (ii) return the identity because it would be invoked on a detached tensor outside the kernel's graph.

The cheapest realistic substitute is **spectral normalization on the single `nn.Parameter` that drives the recurrence**. The chaos step is `b_{i+1} = tanh(W·b_i) + α·sin(β·b_i + φ)`. With `|tanh'| ≤ 1` and `|cos| ≤ 1`, a sufficient Lipschitz bound on `f` is `σ_max(W) + α·β`. Controlling `σ_max(W)` (and clamping `α·β`) directly bounds the spectral radius of the iteration Jacobian and is what the Bai 2019/2021 DEQ guidance actually needs in practice — Miyato 2018 power-iteration SN is the standard way.

## Architecture facts verified

- **The `W` matrix.** Not a `nn.Parameter` on its own — it is a **slice** of `Block.proj.weight`:
  - `test_vortex_2k.py:635` → `self.proj = CastedLinear(dim, dim, bias=True)` (`nn.Linear`-like, `dim = model_dim = 512` default, `num_heads = 8`, so `head_dim = 64`).
  - `test_vortex_2k.py:691-695` carves out `proj_weight_hd = self.proj.weight[:head_dim, :head_dim]` — a `(64, 64)` view handed to the kernel.
  - Inside the kernel, `W` is loaded at `vortex_fused.py:135` and `vortex_fused.py:141-142` and used at `vortex_fused.py:159` as `tl.dot(b_bf16, W)`.
  - `self.proj.weight` is a real `nn.Parameter`. The slice is a torch view so a `torch.nn.utils.parametrize` SN hook on the parent parameter would mangle the non-chaos use of `proj` and is wrong here.
- **Chaos scalars.** `Block.chaos_scalars = nn.Parameter(torch.randn(3))` at `test_vortex_2k.py:647` → `[alpha, beta, phi]`. Regular `nn.Parameter`, accessible outside kernel. Loaded at `vortex_fused.py:144-146`.
- **Depth.** `CHAOS_DEPTH = VORTEX_CHAOS_DEPTH` env var, default 5 (`vortex_fused.py:9`). Recent wins (`67ab894`) use depth with `FWD_NUM_STAGES=3`.
- **Loss site.** Training loss is `loss = model(x, y)` at `test_vortex_2k.py:1126`, backward at `1128`. A scalar regularizer added before `.backward()` would Just Work.

## Why (b) is rejected

A Hutchinson estimator on a PyTorch reference of `chaos_step` requires re-running `b_proj = b @ W ; tanh(...) + α·sin(β·b+φ)` in eager PyTorch. Shape is `(bsz·num_heads, T, head_dim) = (bsz·8, 2048, 64)` per block × `num_layers` blocks. One Rademacher probe is one extra forward+vjp through `CHAOS_DEPTH=5` steps per block per step → easily +15-25% step time even with one probe. That cost has no compose-story with the kernel (we'd lose fwd/bwd fusion), and crawler recurrence (−0.004 BPB/loop) means the regularizer has to be cheap to not cannibalize its own budget. Not worth it.

## Why (c) is deferred

Option (c) — fold `‖J‖_F²` accumulation into `_vortex_chaos_bwd_kernel` using the `dZ = dB * (1 - T_i²)` values it already computes (`vortex_bwd.py:336-397`) — is in principle free: those are the diagonal of the tanh Jacobian. But it adds a per-block `tl.sum(dZ*dZ)` atomic write and a new output tensor, complicating the autograd Function signature and requiring us to re-measure all the sweep champions. One kernel surgery per stability experiment is too much. Revisit after (a) gives signal.

## (a) Concrete design — spectral normalization on `self.proj.weight[:head_dim,:head_dim]`

Gated by env var `SPECTRAL_NORM_W=1` (default off, noop path identical to current behavior).

Insert in `Block.__init__` at `test_vortex_2k.py:635`, and intercept in `Block.forward` at `test_vortex_2k.py:691-706`:

```python
# __init__ additions
self._spectral_norm_w = os.environ.get("SPECTRAL_NORM_W", "0") == "1"
if self._spectral_norm_w:
    head_dim = dim // num_heads
    # One left / right singular vector per Block, persistent across steps.
    self.register_buffer("_sn_u", F.normalize(torch.randn(head_dim), dim=0), persistent=False)
    # Target spectral radius — default 0.95 keeps tanh(W·b) strictly contractive.
    self._sn_sigma_target = float(os.environ.get("SPECTRAL_NORM_SIGMA", "0.95"))
    self._sn_n_power = int(os.environ.get("SPECTRAL_NORM_POWER_ITERS", "1"))

# forward — replace the current slice+contiguous block at line 691-706
if self._spectral_norm_w:
    W_hd = self.proj.weight[:self.attn.head_dim, :self.attn.head_dim]
    # Power iteration on a DETACHED copy to estimate sigma; the gradient still
    # flows through the final divide because we rebuild proj_weight_hd from the
    # live weight, not the detached one.
    with torch.no_grad():
        u = self._sn_u
        Wd = W_hd.detach()
        for _ in range(self._sn_n_power):
            v = F.normalize(Wd.t() @ u, dim=0, eps=1e-8)
            u = F.normalize(Wd @ v, dim=0, eps=1e-8)
        self._sn_u.copy_(u)
    sigma = torch.einsum("i,ij,j->", u, W_hd, v)          # differentiable
    scale = torch.clamp(self._sn_sigma_target / (sigma.abs() + 1e-6), max=1.0)
    proj_weight_hd = (W_hd * scale).contiguous()
else:
    proj_weight_hd = self.proj.weight[:self.attn.head_dim, :self.attn.head_dim].contiguous()
```

Key notes:
- `sigma` is a scalar that **participates in the autograd graph via `W_hd`**, so the kernel's internal `dW` flows back correctly (SN only rescales; `clamp(…, max=1.0)` means when `sigma < target` we pass `W` through unchanged, matching the Miyato "divide only when violated" variant).
- We only rescale the `(head_dim, head_dim)` corner. The rest of `proj.weight` (used for Stream B `dim→dim` projection downstream? — actually the proj here is not used elsewhere in this Block; `self.proj` is only consumed via the slice on line 693). So a full-matrix SN (`F.normalize(W_full)` via `torch.nn.utils.spectral_norm`) would also work and is simpler — but only if the full `self.proj.weight` is in fact nowhere else referenced. Quick grep confirms `self.proj` in `Block` is referenced **only** at line 691-695, so `torch.nn.utils.parametrizations.spectral_norm(self.proj)` is a legal one-liner alternative. I prefer the explicit-buffer version above because it keeps `sigma` observable for telemetry.

### Cost budget (head_dim=64, 12 layers @ model_dim=512)

- Power iter: two `64×64 @ 64` matvecs + two `normalize` = ~2·64² = 8192 flops per iter per block. 12 blocks × 1 iter = ~0.1 µs on an H100. **Negligible.**
- Extra memory: `_sn_u` buffer of shape `(head_dim,)` per block = 64 fp32 × 12 blocks = 3 KB. **Free.**
- Autograd overhead from `einsum` + `clamp` + scalar multiply: one tiny kernel launch per block per step, <10 µs aggregate. At 300 steps total this is sub-second.

### Expected signal

- **Good**: −0.001 to −0.004 val_bpb by letting the chaos loop actually settle instead of fighting divergence. Rationale: "crawler recurrence validated" memo says each chaos loop adds −0.004 to −0.005 when recurrence is stable; if current `CHAOS_DEPTH=5` is partially undone by `W` eigenvalues >1, SN reclaims that.
- **Flat/worse**: +0.000 to +0.005. Means `W` is already well-behaved at Muon-trained scale, or the `α·β` term dominates and `W`-SN is attacking the wrong axis. In that case the next move is to **also** clamp `α` via `α.data.clamp_(-0.5, 0.5)` after each step.
- **Hard fail threshold**: +0.010 or worse = kill, same threshold used for Helix/Ouro closures.

### Validation — prove the regularizer is doing something

Log, every 25 steps, to the same line that already emits `val_loss` at `test_vortex_2k.py:1102`:

```python
if step % 25 == 0 and args.rank == 0:
    with torch.no_grad():
        sigmas = []
        for blk in model.blocks:
            if getattr(blk, "_spectral_norm_w", False):
                W_hd = blk.proj.weight[:blk.attn.head_dim, :blk.attn.head_dim]
                # one-shot top-SV via torch.linalg.svdvals; tiny on 64x64
                s = torch.linalg.svdvals(W_hd.float())[0].item()
                sigmas.append(s)
        log0(f"step:{step} sigma_max_mean:{sum(sigmas)/len(sigmas):.4f} "
             f"sigma_max_worst:{max(sigmas):.4f}")
```

- **Signature of success**: `sigma_max` hovers at or just below `SPECTRAL_NORM_SIGMA` (0.95) across training. If it's pinned at 0.95 and `scale < 1.0` is being applied, the regularizer is actively biting. If it floats freely at 0.3, SN is a noop and we have our answer (no stability problem → this knob is dead).
- **Signature of failure**: `sigma_max` oscillates or the `val_bpb` plot has the same shape as baseline but shifted up → regularizer hurts without reason. Kill.

## Gate suggestion

Run one 4xGPU (per `feedback_production_defaults.md` gate convention) sweep:
- `SPECTRAL_NORM_W=1 SPECTRAL_NORM_SIGMA=0.95` (primary)
- `SPECTRAL_NORM_W=1 SPECTRAL_NORM_SIGMA=0.80` (aggressive)
- `SPECTRAL_NORM_W=0` (baseline) — already the current champion commit `67ab894`.

300 steps each. Same seed sweep as BW21. Promote to 8xH100 only if primary clears −0.002 vs baseline (same bar as BW21 NoisyQAT).

## Honest caveat

Before spending the gate budget, re-examine `project_helix_status.md`: Helix already closed with +0.140 on a micro→scale mismatch. Spectral norm is a different axis (it stabilizes rather than augments), but the prior on "chaos-loop-tuning knobs move val_bpb" is weak. If 300-step sweep shows ±0.001, don't escalate — close this question and move the budget to megakernel work per `feedback_megakernel_first.md`.
