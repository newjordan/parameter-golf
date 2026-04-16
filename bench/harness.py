"""Timing + memory utilities for the Vortex bench."""
from __future__ import annotations

import contextlib
import dataclasses
import statistics
from typing import Callable

import torch


@dataclasses.dataclass
class BenchResult:
    name: str
    fwd_ms: float
    bwd_ms: float
    fwd_p50: float
    bwd_p50: float
    peak_mem_mb: float
    iters: int
    ok: bool
    err: str = ""


def _cuda_sync():
    torch.cuda.synchronize()


def _reset_mem():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def time_one(fn: Callable[[], torch.Tensor], iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    _cuda_sync()
    ts: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        ts.append(start.elapsed_time(end))
    return ts


def bench_fwd_bwd(
    name: str,
    build_inputs: Callable[[], tuple],
    fwd_fn: Callable[..., torch.Tensor],
    *,
    iters: int = 20,
    warmup: int = 5,
) -> BenchResult:
    """Run fwd+bwd timing. `build_inputs()` must return a tuple whose first
    element is a list of leaf tensors to accumulate grads on, followed by
    the arguments passed to `fwd_fn`.
    """
    try:
        _reset_mem()
        leaves, args = build_inputs()
        # Forward-only timing
        def _fwd():
            with torch.no_grad():
                return fwd_fn(*args)
        fwd_ts = time_one(_fwd, iters=iters, warmup=warmup)

        # Forward+backward timing (fresh graph every iter)
        def _fwd_bwd():
            for t in leaves:
                if t.grad is not None:
                    t.grad = None
            out = fwd_fn(*args)
            go = torch.ones_like(out)
            out.backward(go)
        bwd_ts = time_one(_fwd_bwd, iters=iters, warmup=warmup)

        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return BenchResult(
            name=name,
            fwd_ms=statistics.mean(fwd_ts),
            bwd_ms=statistics.mean(bwd_ts),
            fwd_p50=statistics.median(fwd_ts),
            bwd_p50=statistics.median(bwd_ts),
            peak_mem_mb=peak,
            iters=iters,
            ok=True,
        )
    except Exception as e:
        return BenchResult(
            name=name, fwd_ms=float("nan"), bwd_ms=float("nan"),
            fwd_p50=float("nan"), bwd_p50=float("nan"), peak_mem_mb=0.0,
            iters=0, ok=False, err=f"{type(e).__name__}: {e}",
        )


TSV_HEADER = "\t".join([
    "impl", "B", "H", "T", "D", "dtype",
    "fwd_ms", "fwd_p50_ms", "bwd_ms", "bwd_p50_ms",
    "peak_mem_mb", "iters", "ok", "err",
])


def tsv_row(r: BenchResult, B: int, H: int, T: int, D: int, dtype: str) -> str:
    return "\t".join([
        r.name, str(B), str(H), str(T), str(D), dtype,
        f"{r.fwd_ms:.4f}", f"{r.fwd_p50:.4f}",
        f"{r.bwd_ms:.4f}", f"{r.bwd_p50:.4f}",
        f"{r.peak_mem_mb:.2f}", str(r.iters),
        "1" if r.ok else "0", r.err.replace("\t", " ")[:200],
    ])


@contextlib.contextmanager
def autocast_none():
    yield
