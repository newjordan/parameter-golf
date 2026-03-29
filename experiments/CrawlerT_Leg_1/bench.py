#!/usr/bin/env python3
"""
CrawlerT_Leg_1 — bench.py
Phase 1: Does torch.compile(max-autotune) speed up the crawler's NormMLP hot path?

Run on a single GPU (no torchrun needed):
    python bench.py

Config mirrors Crawler_Leg_1:
    dim=512, mlp_mult=4.0, B=48, T=2048 (per-GPU batch on 8xH100)
"""
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Minimal reimplementation of the relevant classes ──────────────────────────
# (standalone — no dependency on train_gpt.py or flash_attn)

class CastedLinear(nn.Linear):
    """fp32 weights, cast to input dtype on each forward — mirrors train_gpt.py."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


class RMSNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),))


class MLP(nn.Module):
    """relu_sq MLP — mirrors train_gpt.py exactly."""
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.relu(x)
        return self.proj(x.square())


class NormMLP(nn.Module):
    """
    The MLP branch of Block.forward, isolated:
        mlp_scale * MLP(RMSNorm(x) * ln_scale_factor)
    This is what runs 4x per crawler loop (K=4).
    """
    def __init__(self, dim: int, mlp_mult: float, ln_scale_factor: float = 1.0):
        super().__init__()
        self.norm = RMSNorm()
        self.mlp = MLP(dim, mlp_mult)
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.ln_scale_factor = ln_scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x) * self.ln_scale_factor
        return self.mlp_scale.to(x.dtype)[None, None, :] * self.mlp(normed)


# ── Benchmark harness ─────────────────────────────────────────────────────────

def bench_fwd(fn, x: torch.Tensor, label: str, n_warmup: int = 30, n_iter: int = 300) -> float:
    for _ in range(n_warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn(x)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iter * 1000
    print(f"    {label:<45}: {ms:.4f} ms/call")
    return ms


def bench_fwd_bwd(fn, x: torch.Tensor, label: str, n_warmup: int = 20, n_iter: int = 200) -> float:
    for _ in range(n_warmup):
        out = fn(x)
        out.sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = fn(x)
        out.sum().backward()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iter * 1000
    print(f"    {label:<45}: {ms:.4f} ms/call")
    return ms


def make_model(dim: int, mlp_mult: float, device: str) -> NormMLP:
    m = NormMLP(dim, mlp_mult)
    m = m.to(device=device)  # weights stay fp32
    return m


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run on a GPU node.")
        return

    device = "cuda"
    dtype = torch.bfloat16

    # Crawler_Leg_1 config: dim=512, mlp_mult=4.0 (crawler), 8xH100 → 48 seqs/GPU
    DIM = 512
    MLP_MULT = 4.0
    B, T = 48, 2048

    print()
    print("=" * 65)
    print("  CrawlerT_Leg_1 — NormMLP Kernel Fusion Benchmark")
    print(f"  Config: dim={DIM}, mlp_mult={MLP_MULT}, B={B}, T={T}, dtype={dtype}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    x = torch.randn(B, T, DIM, device=device, dtype=dtype, requires_grad=True)

    # ── Test A: Eager (baseline) ─────────────────────────────────────────────
    print("\n[A] Eager (baseline — equivalent to current Crawler_Leg_1):")
    ma = make_model(DIM, MLP_MULT, device)
    ma.eval()
    t_a_fwd = bench_fwd(lambda inp: ma(inp), x, "forward")

    ma.train()
    x_g = x.detach().requires_grad_(True)
    t_a_fwdbwd = bench_fwd_bwd(lambda inp: ma(inp), x_g, "forward + backward")

    # ── Test B: compile(mode=None) — current default ─────────────────────────
    print("\n[B] torch.compile(mode=None) — current training default:")
    mb = make_model(DIM, MLP_MULT, device)
    mb.eval()
    mb_compiled = torch.compile(mb, dynamic=False, fullgraph=False)
    print("    (compiling — first call will be slow)...")
    mb_compiled(x)  # trigger compile
    torch.cuda.synchronize()
    t_b_fwd = bench_fwd(lambda inp: mb_compiled(inp), x, "forward")

    mb.train()
    mb_tr_compiled = torch.compile(mb, dynamic=False, fullgraph=False)
    x_g = x.detach().requires_grad_(True)
    mb_tr_compiled(x_g).sum().backward()  # trigger compile
    torch.cuda.synchronize()
    t_b_fwdbwd = bench_fwd_bwd(lambda inp: mb_tr_compiled(inp), x_g, "forward + backward")

    # ── Test C: compile(mode='max-autotune') ─────────────────────────────────
    print("\n[C] torch.compile(mode='max-autotune') — proposed TRITON_FUSE=1:")
    mc = make_model(DIM, MLP_MULT, device)
    mc.eval()
    mc_compiled = torch.compile(mc, dynamic=False, fullgraph=False, mode="max-autotune")
    print("    (compiling with max-autotune — may take 60-120s on first run)...")
    mc_compiled(x)  # trigger compile
    torch.cuda.synchronize()
    t_c_fwd = bench_fwd(lambda inp: mc_compiled(inp), x, "forward")

    mc.train()
    mc_tr_compiled = torch.compile(mc, dynamic=False, fullgraph=False, mode="max-autotune")
    x_g = x.detach().requires_grad_(True)
    mc_tr_compiled(x_g).sum().backward()  # trigger compile
    torch.cuda.synchronize()
    t_c_fwdbwd = bench_fwd_bwd(lambda inp: mc_tr_compiled(inp), x_g, "forward + backward")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'':45}  {'fwd':>8}  {'fwd+bwd':>8}")
    print(f"  {'A: Eager (baseline)':45}  {t_a_fwd:>7.3f}ms  {t_a_fwdbwd:>7.3f}ms")
    print(f"  {'B: compile(default)':45}  {t_b_fwd:>7.3f}ms  {t_b_fwdbwd:>7.3f}ms  "
          f"[{t_a_fwd/t_b_fwd:.2f}x / {t_a_fwdbwd/t_b_fwdbwd:.2f}x]")
    print(f"  {'C: compile(max-autotune)':45}  {t_c_fwd:>7.3f}ms  {t_c_fwdbwd:>7.3f}ms  "
          f"[{t_a_fwd/t_c_fwd:.2f}x / {t_a_fwdbwd/t_c_fwdbwd:.2f}x]")
    print()

    # Decision
    speedup_fwd = t_a_fwd / t_c_fwd
    speedup_fwdbwd = t_a_fwdbwd / t_c_fwdbwd

    print("  DECISION:")
    if speedup_fwdbwd >= 1.15:
        print(f"  ✓ PROCEED to Phase 2. max-autotune gives {(speedup_fwdbwd-1)*100:.1f}% speedup (fwd+bwd).")
        print("    → Run: TRITON_FUSE=1 bash run.sh")
    elif speedup_fwdbwd >= 1.10:
        print(f"  ~ MARGINAL. max-autotune gives {(speedup_fwdbwd-1)*100:.1f}% speedup (fwd+bwd).")
        print("    → Worth a full run to confirm BPB delta.")
        print("    → Run: TRITON_FUSE=1 bash run.sh")
    else:
        print(f"  ✗ NO SIGNAL. max-autotune gives only {(speedup_fwdbwd-1)*100:.1f}% speedup.")
        print("    → torch.compile is not the bottleneck here.")
        print("    → Investigate: CastedLinear weight-cast overhead, custom Triton fused_relu_sq kernel.")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
