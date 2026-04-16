"""Autoresearch driver: sweep kernel config / seqlen / chaos via subprocesses.

Each sub-run is a cold-cache launch (fresh triton JIT) which mirrors
production behavior and avoids cache pollution between configs.

Usage:
    python -m bench.autoresearch --mode kernel_cfg --B 4 --H 8 --T 2048 \\
        --out records/autores_kernel_cfg.tsv

Modes:
    kernel_cfg  sweep (BLOCK_SIZE, fwd_warps, fwd_stages, bwd_chaos_warps,
                bwd_chaos_stages, bwd_attn_warps, bwd_attn_stages)
    seqlen      sweep T in a list at fixed cfg
    chaos       sweep chaos_iters end-to-end: VORTEX_CHAOS_DEPTH re-compiles
                the fused kernel at the requested depth, and the eager/sdpa
                references track the same depth for apples-to-apples numbers.
                Depths are clamped to [0, 7] by the kernel constexpr ladder.
"""
from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
import threading
import time

HEADER = (
    "impl\tB\tH\tT\tD\tdtype\tfwd_ms\tfwd_p50_ms\tbwd_ms\tbwd_p50_ms\t"
    "peak_mem_mb\titers\tok\terr\t"
    "cfg_block\tcfg_fwd_warps\tcfg_fwd_stages\t"
    "cfg_bwd_chaos_warps\tcfg_bwd_chaos_stages\t"
    "cfg_bwd_attn_warps\tcfg_bwd_attn_stages\tcfg_chaos_iters\n"
)

_write_lock = threading.Lock()


def run_one(env_overrides: dict, cli_args: list[str], gpu_id: int | None = None) -> list[str]:
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [sys.executable, "-m", "bench.vortex_bench"] + cli_args
    gpu_tag = f"gpu{gpu_id}" if gpu_id is not None else "gpu?"
    print(f"# [{gpu_tag}] $ {' '.join(shlex.quote(c) for c in cmd)}  env={env_overrides}", flush=True)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    rows = []
    for line in p.stdout.splitlines():
        if not line or line.startswith("#") or line.startswith("impl\t"):
            continue
        rows.append(line)
    if p.returncode != 0:
        print(f"# [{gpu_tag}] FAILED rc={p.returncode}", file=sys.stderr)
        print(p.stderr[-2000:], file=sys.stderr)
    return rows


def _resolve_cfg(cfg: dict) -> dict:
    """Merge process env with per-run overrides so the TSV shows what actually ran."""
    return {
        "VORTEX_BLOCK_SIZE": cfg.get("VORTEX_BLOCK_SIZE", os.environ.get("VORTEX_BLOCK_SIZE", 64)),
        "VORTEX_FWD_NUM_WARPS": cfg.get("VORTEX_FWD_NUM_WARPS", os.environ.get("VORTEX_FWD_NUM_WARPS", 4)),
        "VORTEX_FWD_NUM_STAGES": cfg.get("VORTEX_FWD_NUM_STAGES", os.environ.get("VORTEX_FWD_NUM_STAGES", 1)),
        "VORTEX_BWD_CHAOS_NUM_WARPS": cfg.get("VORTEX_BWD_CHAOS_NUM_WARPS", os.environ.get("VORTEX_BWD_CHAOS_NUM_WARPS", 4)),
        "VORTEX_BWD_CHAOS_NUM_STAGES": cfg.get("VORTEX_BWD_CHAOS_NUM_STAGES", os.environ.get("VORTEX_BWD_CHAOS_NUM_STAGES", 1)),
        "VORTEX_BWD_ATTN_NUM_WARPS": cfg.get("VORTEX_BWD_ATTN_NUM_WARPS", os.environ.get("VORTEX_BWD_ATTN_NUM_WARPS", 4)),
        "VORTEX_BWD_ATTN_NUM_STAGES": cfg.get("VORTEX_BWD_ATTN_NUM_STAGES", os.environ.get("VORTEX_BWD_ATTN_NUM_STAGES", 1)),
        "CHAOS_ITERS": cfg.get("CHAOS_ITERS", os.environ.get("VORTEX_CHAOS_DEPTH", 5)),
    }


def _append(out_path: str, cfg: dict, rows: list[str]):
    resolved = _resolve_cfg(cfg)
    cfg_cols = "\t".join(str(resolved[k]) for k in (
        "VORTEX_BLOCK_SIZE", "VORTEX_FWD_NUM_WARPS", "VORTEX_FWD_NUM_STAGES",
        "VORTEX_BWD_CHAOS_NUM_WARPS", "VORTEX_BWD_CHAOS_NUM_STAGES",
        "VORTEX_BWD_ATTN_NUM_WARPS", "VORTEX_BWD_ATTN_NUM_STAGES",
        "CHAOS_ITERS",
    ))
    with _write_lock:
        exists = os.path.exists(out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "a") as f:
            if not exists:
                f.write(HEADER)
            for r in rows:
                f.write(r + "\t" + cfg_cols + "\n")


def _dispatch_parallel(args, jobs: list[tuple[dict, list[str]]]):
    """Run (env, cli) jobs across args.num_gpus workers using a shared queue."""
    n_gpus = max(1, args.num_gpus)
    import queue as _q
    q: _q.Queue = _q.Queue()
    for i, job in enumerate(jobs):
        q.put((i, job))
    total = len(jobs)

    def _worker(gpu_id: int):
        while True:
            try:
                idx, (env, cli) = q.get_nowait()
            except _q.Empty:
                return
            t0 = time.time()
            rows = run_one(env, cli, gpu_id=gpu_id)
            rows = [r for r in rows if r.startswith("vortex_fused\t") or args.keep_all_rows]
            _append(args.out, env, rows)
            print(f"# [{idx+1}/{total} gpu{gpu_id}] done in {time.time()-t0:.1f}s", flush=True)

    ts = [threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(n_gpus)]
    for t in ts: t.start()
    for t in ts: t.join()


def mode_kernel_cfg(args):
    # Sweep the kernel config grid. Keep cost bounded by not exploding shapes.
    block_sizes = args.block_sizes or [32, 64, 128]
    fwd_warps = args.fwd_warps or [2, 4, 8]
    fwd_stages = args.fwd_stages or [1, 2, 3]
    bwd_warps = args.bwd_warps or [2, 4, 8]
    bwd_stages = args.bwd_stages or [1, 2]
    jobs = []
    for bs, fw, fs, bw, bs2 in itertools.product(block_sizes, fwd_warps, fwd_stages, bwd_warps, bwd_stages):
        env = {
            "VORTEX_BLOCK_SIZE": bs,
            "VORTEX_FWD_NUM_WARPS": fw,
            "VORTEX_FWD_NUM_STAGES": fs,
            "VORTEX_BWD_CHAOS_NUM_WARPS": bw,
            "VORTEX_BWD_CHAOS_NUM_STAGES": bs2,
            "VORTEX_BWD_ATTN_NUM_WARPS": bw,
            "VORTEX_BWD_ATTN_NUM_STAGES": bs2,
        }
        cli = ["--B", str(args.B), "--H", str(args.H), "--T", str(args.T), "--D", str(args.D),
               "--iters", str(args.iters), "--warmup", str(args.warmup),
               "--chaos-iters", str(args.chaos_iters), "--fused-only"]
        jobs.append((env, cli))
    _dispatch_parallel(args, jobs)


def mode_bwd_focus(args):
    """Decoupled bwd sweep: VORTEX_BWD_CHAOS_{WARPS,STAGES} and
    VORTEX_BWD_ATTN_{WARPS,STAGES} vary independently. fwd config is pinned
    (via env or --fwd-warps/--fwd-stages/--block-sizes) so we isolate the bwd.
    """
    block_sizes = args.block_sizes or [64, 128]
    fwd_warps = args.fwd_warps or [int(os.environ.get("VORTEX_FWD_NUM_WARPS", 8))]
    fwd_stages = args.fwd_stages or [int(os.environ.get("VORTEX_FWD_NUM_STAGES", 2))]
    bwd_chaos_warps = args.bwd_chaos_warps or [2, 4, 8]
    bwd_chaos_stages = args.bwd_chaos_stages or [1, 2, 3]
    bwd_attn_warps = args.bwd_attn_warps or [2, 4, 8]
    bwd_attn_stages = args.bwd_attn_stages or [1, 2, 3]
    jobs = []
    for bs, fw, fs, bcw, bcs, baw, bas in itertools.product(
        block_sizes, fwd_warps, fwd_stages,
        bwd_chaos_warps, bwd_chaos_stages, bwd_attn_warps, bwd_attn_stages
    ):
        env = {
            "VORTEX_BLOCK_SIZE": bs,
            "VORTEX_FWD_NUM_WARPS": fw,
            "VORTEX_FWD_NUM_STAGES": fs,
            "VORTEX_BWD_CHAOS_NUM_WARPS": bcw,
            "VORTEX_BWD_CHAOS_NUM_STAGES": bcs,
            "VORTEX_BWD_ATTN_NUM_WARPS": baw,
            "VORTEX_BWD_ATTN_NUM_STAGES": bas,
            "VORTEX_CHAOS_DEPTH": args.chaos_iters,
        }
        cli = ["--B", str(args.B), "--H", str(args.H), "--T", str(args.T), "--D", str(args.D),
               "--iters", str(args.iters), "--warmup", str(args.warmup),
               "--chaos-iters", str(args.chaos_iters), "--fused-only"]
        jobs.append((env, cli))
    print(f"# bwd_focus: {len(jobs)} configs on {args.num_gpus} GPU(s)", flush=True)
    _dispatch_parallel(args, jobs)


def mode_seqlen(args):
    seqlens = args.seqlens or [512, 1024, 2048, 4096, 8192]
    env = {}  # default cfg (or caller-pinned via VORTEX_* envs)
    for t in seqlens:
        cli = ["--B", str(args.B), "--H", str(args.H), "--T", str(t), "--D", str(args.D),
               "--iters", str(args.iters), "--warmup", str(args.warmup),
               "--chaos-iters", str(args.chaos_iters)]
        rows = run_one(env, cli)
        _append(args.out, env, rows)


def mode_chaos(args):
    iters_list = args.chaos_iters_list or [0, 1, 3, 5, 7]
    for ci in iters_list:
        cli = ["--B", str(args.B), "--H", str(args.H), "--T", str(args.T), "--D", str(args.D),
               "--iters", str(args.iters), "--warmup", str(args.warmup),
               "--chaos-iters", str(ci)]
        # VORTEX_CHAOS_DEPTH parameterises the fused kernel; CHAOS_ITERS is
        # retained for legacy dashboards. Both track the eager chaos depth.
        env = {"CHAOS_ITERS": ci, "VORTEX_CHAOS_DEPTH": ci}
        rows = run_one(env, cli)
        _append(args.out, env, rows)


def mode_t_depth(args):
    """Effective-potential surface: sweep T x depth at a single pinned kernel
    config (passed via env knobs + --block-sizes/--fwd-warps/... single values).
    Emits baselines too so downstream reporting can compute speedups."""
    seqlens = args.seqlens or [512, 1024, 2048, 4096, 8192]
    iters_list = args.chaos_iters_list or [0, 1, 3, 5, 7]
    pinned = {}
    if args.block_sizes and len(args.block_sizes) == 1:
        pinned["VORTEX_BLOCK_SIZE"] = args.block_sizes[0]
    if args.fwd_warps and len(args.fwd_warps) == 1:
        pinned["VORTEX_FWD_NUM_WARPS"] = args.fwd_warps[0]
    if args.fwd_stages and len(args.fwd_stages) == 1:
        pinned["VORTEX_FWD_NUM_STAGES"] = args.fwd_stages[0]
    if args.bwd_chaos_warps and len(args.bwd_chaos_warps) == 1:
        pinned["VORTEX_BWD_CHAOS_NUM_WARPS"] = args.bwd_chaos_warps[0]
    if args.bwd_chaos_stages and len(args.bwd_chaos_stages) == 1:
        pinned["VORTEX_BWD_CHAOS_NUM_STAGES"] = args.bwd_chaos_stages[0]
    if args.bwd_attn_warps and len(args.bwd_attn_warps) == 1:
        pinned["VORTEX_BWD_ATTN_NUM_WARPS"] = args.bwd_attn_warps[0]
    if args.bwd_attn_stages and len(args.bwd_attn_stages) == 1:
        pinned["VORTEX_BWD_ATTN_NUM_STAGES"] = args.bwd_attn_stages[0]
    jobs = []
    for t in seqlens:
        for ci in iters_list:
            env = dict(pinned)
            env["CHAOS_ITERS"] = ci
            env["VORTEX_CHAOS_DEPTH"] = ci
            cli = ["--B", str(args.B), "--H", str(args.H), "--T", str(t), "--D", str(args.D),
                   "--iters", str(args.iters), "--warmup", str(args.warmup),
                   "--chaos-iters", str(ci)]
            jobs.append((env, cli))
    print(f"# t_depth: {len(jobs)} configs on {args.num_gpus} GPU(s)", flush=True)
    _dispatch_parallel(args, jobs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["kernel_cfg", "seqlen", "chaos", "bwd_focus", "t_depth"], required=True)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--chaos-iters", type=int, default=5, dest="chaos_iters")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--num-gpus", type=int, default=1, help="Parallel GPU workers (each launch pins CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--keep-all-rows", action="store_true", help="Keep baseline rows in TSV (default keeps only vortex_fused)")
    ap.add_argument("--block-sizes", type=int, nargs="*")
    ap.add_argument("--fwd-warps", type=int, nargs="*")
    ap.add_argument("--fwd-stages", type=int, nargs="*")
    ap.add_argument("--bwd-warps", type=int, nargs="*", help="(kernel_cfg) tied bwd_chaos/bwd_attn warps")
    ap.add_argument("--bwd-stages", type=int, nargs="*", help="(kernel_cfg) tied bwd_chaos/bwd_attn stages")
    ap.add_argument("--bwd-chaos-warps", type=int, nargs="*", dest="bwd_chaos_warps")
    ap.add_argument("--bwd-chaos-stages", type=int, nargs="*", dest="bwd_chaos_stages")
    ap.add_argument("--bwd-attn-warps", type=int, nargs="*", dest="bwd_attn_warps")
    ap.add_argument("--bwd-attn-stages", type=int, nargs="*", dest="bwd_attn_stages")
    ap.add_argument("--seqlens", type=int, nargs="*")
    ap.add_argument("--chaos-iters-list", type=int, nargs="*", dest="chaos_iters_list")
    args = ap.parse_args()
    t0 = time.time()
    if args.mode == "kernel_cfg":
        mode_kernel_cfg(args)
    elif args.mode == "seqlen":
        mode_seqlen(args)
    elif args.mode == "chaos":
        mode_chaos(args)
    elif args.mode == "bwd_focus":
        mode_bwd_focus(args)
    elif args.mode == "t_depth":
        mode_t_depth(args)
    print(f"# autoresearch total wall: {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
