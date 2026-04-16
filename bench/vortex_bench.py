"""Single-GPU vortex-fused bench: compare against eager/SDPA baselines.

Usage:
    python -m bench.vortex_bench --B 2 --H 8 --T 2048 --D 128 --iters 20
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from torch.nn.attention import SDPBackend

from bench.harness import BenchResult, bench_fwd_bwd, tsv_row, TSV_HEADER
from bench import impls


BASELINE_SDPA = [
    ("sdpa_flash_attn_only", SDPBackend.FLASH_ATTENTION, False),
    ("sdpa_cudnn_attn_only", SDPBackend.CUDNN_ATTENTION, False),
    ("sdpa_memeff_attn_only", SDPBackend.EFFICIENT_ATTENTION, False),
    ("sdpa_flash_vortex", SDPBackend.FLASH_ATTENTION, True),
    ("sdpa_cudnn_vortex", SDPBackend.CUDNN_ATTENTION, True),
]


def run(args) -> list[BenchResult]:
    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    B, H, T, D = args.B, args.H, args.T, args.D
    it, wu = args.iters, args.warmup
    results: list[BenchResult] = []

    if not args.fused_only:
        # Attention-only baselines
        for name, backend, _ in BASELINE_SDPA:
            if not name.endswith("attn_only"):
                continue
            def _build(backend=backend):
                leaves, fwd, xargs = impls.build_sdpa_attn_only(B, H, T, D, device, backend=backend, dtype=dtype)
                return leaves, xargs
            def _fwd(q, k, v, backend=backend):
                import torch.nn.functional as F
                from torch.nn.attention import sdpa_kernel
                with sdpa_kernel(backend):
                    return F.scaled_dot_product_attention(q, k, v, is_causal=True)
            results.append(bench_fwd_bwd(name, _build, _fwd, iters=it, warmup=wu))

        # Vortex-shaped baselines (SDPA attention + eager chaos stream)
        for name, backend, is_vortex in BASELINE_SDPA:
            if not is_vortex:
                continue
            def _build(backend=backend):
                leaves, fwd, xargs = impls.build_sdpa_vortex(B, H, T, D, device, backend=backend, dtype=dtype, chaos_iters=args.chaos_iters)
                return leaves, (fwd, *xargs)
            def _fwd(fwd, *xa):
                return fwd(*xa)
            results.append(bench_fwd_bwd(name, _build, _fwd, iters=it, warmup=wu))

        # Eager reference (fp32 internal), at same chaos depth
        def _build_eager():
            leaves, fwd, xargs = impls.build_eager(B, H, T, D, device, dtype=dtype, chaos_iters=args.chaos_iters)
            return leaves, (fwd, *xargs)
        def _fwd_eager(fwd, *xa):
            return fwd(*xa)
        results.append(bench_fwd_bwd("eager_vortex", _build_eager, _fwd_eager, iters=it, warmup=wu))

    # Fused vortex kernel
    def _build_vortex():
        leaves, fwd, xargs = impls.build_vortex_fused(B, H, T, D, device, dtype=dtype)
        return leaves, (fwd, *xargs)
    def _fwd_vortex(fwd, *xa):
        return fwd(*xa)
    results.append(bench_fwd_bwd("vortex_fused", _build_vortex, _fwd_vortex, iters=it, warmup=wu))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--chaos-iters", type=int, default=5, dest="chaos_iters")
    ap.add_argument("--out", type=str, default=None, help="TSV output path (append)")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--fused-only", action="store_true", help="Skip SDPA/eager baselines (for config sweeps)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        sys.exit(1)

    dev = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"# device: {dev} sm_{cc[0]}{cc[1]}  torch={torch.__version__}")
    print(f"# shape: B={args.B} H={args.H} T={args.T} D={args.D} dtype={args.dtype} chaos_iters={args.chaos_iters}")
    print(f"# tag: {args.tag}")
    print(TSV_HEADER)

    t0 = time.time()
    results = run(args)
    dt = time.time() - t0
    rows = []
    for r in results:
        row = tsv_row(r, args.B, args.H, args.T, args.D, args.dtype)
        rows.append(row)
        print(row)
    print(f"# total_wall_s: {dt:.1f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        exists = os.path.exists(args.out)
        with open(args.out, "a") as f:
            if not exists:
                f.write(TSV_HEADER + "\n")
            for row in rows:
                f.write(row + "\n")


if __name__ == "__main__":
    main()
