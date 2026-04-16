"""Summarize autoresearch TSVs into a markdown report.

Usage:
    python -m bench.report --kernel_cfg records/autores_kernel_cfg.tsv \\
        --seqlen records/autores_seqlen.tsv --chaos records/autores_chaos.tsv \\
        --out records/autores_summary.md
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Any


def _read_tsv(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def _fmt_float(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)


def _best_fwd(rows: list[dict]) -> dict:
    return min((r for r in rows if r.get("ok") == "1"), key=lambda r: float(r["fwd_ms"]))


def _best_bwd(rows: list[dict]) -> dict:
    return min((r for r in rows if r.get("ok") == "1"), key=lambda r: float(r["bwd_ms"]))


def _section_kernel_cfg(rows: list[dict]) -> str:
    if not rows:
        return ""
    out = ["## Kernel config sweep (vortex_fused)\n"]
    out.append(f"Configs evaluated: {len(rows)}")
    try:
        r = _best_fwd(rows)
        out.append(f"- **Best fwd_ms** = {_fmt_float(r['fwd_ms'])} ms @ "
                   f"block={r['cfg_block']}, fwd_warps={r['cfg_fwd_warps']}, "
                   f"fwd_stages={r['cfg_fwd_stages']}")
        r = _best_bwd(rows)
        out.append(f"- **Best bwd_ms** = {_fmt_float(r['bwd_ms'])} ms @ "
                   f"block={r['cfg_block']}, bwd_warps={r['cfg_bwd_chaos_warps']}, "
                   f"bwd_stages={r['cfg_bwd_chaos_stages']}")
    except ValueError:
        out.append("- All configs failed.")

    out.append("\nTop 8 by fwd_ms:\n")
    out.append("| block | fw_w | fw_s | bw_w | bw_s | fwd_ms | bwd_ms | mem MB |")
    out.append("|---|---|---|---|---|---|---|---|")
    okrows = [r for r in rows if r.get("ok") == "1"]
    for r in sorted(okrows, key=lambda r: float(r["fwd_ms"]))[:8]:
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["cfg_block"], r["cfg_fwd_warps"], r["cfg_fwd_stages"],
            r["cfg_bwd_chaos_warps"], r["cfg_bwd_chaos_stages"],
            _fmt_float(r["fwd_ms"]), _fmt_float(r["bwd_ms"]),
            _fmt_float(r["peak_mem_mb"], 1)))
    out.append("\nTop 8 by bwd_ms:\n")
    out.append("| block | fw_w | fw_s | bw_w | bw_s | fwd_ms | bwd_ms | mem MB |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(okrows, key=lambda r: float(r["bwd_ms"]))[:8]:
        out.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["cfg_block"], r["cfg_fwd_warps"], r["cfg_fwd_stages"],
            r["cfg_bwd_chaos_warps"], r["cfg_bwd_chaos_stages"],
            _fmt_float(r["fwd_ms"]), _fmt_float(r["bwd_ms"]),
            _fmt_float(r["peak_mem_mb"], 1)))
    return "\n".join(out) + "\n"


def _section_seqlen(rows: list[dict]) -> str:
    if not rows:
        return ""
    out = ["## Seqlen scaling (fwd_ms / bwd_ms, all impls)\n"]
    by_t = {}
    for r in rows:
        by_t.setdefault(r["T"], {})[r["impl"]] = r
    impls = sorted({r["impl"] for r in rows})
    Ts = sorted(by_t.keys(), key=int)
    out.append("### fwd_ms")
    out.append("| T | " + " | ".join(impls) + " |")
    out.append("|---|" + "|".join("---" for _ in impls) + "|")
    for T in Ts:
        cells = []
        for imp in impls:
            r = by_t[T].get(imp)
            cells.append(_fmt_float(r["fwd_ms"]) if (r and r.get("ok") == "1") else "x")
        out.append(f"| {T} | " + " | ".join(cells) + " |")
    out.append("\n### bwd_ms")
    out.append("| T | " + " | ".join(impls) + " |")
    out.append("|---|" + "|".join("---" for _ in impls) + "|")
    for T in Ts:
        cells = []
        for imp in impls:
            r = by_t[T].get(imp)
            cells.append(_fmt_float(r["bwd_ms"]) if (r and r.get("ok") == "1") else "x")
        out.append(f"| {T} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _section_chaos(rows: list[dict]) -> str:
    if not rows:
        return ""
    out = ["## Chaos-stream ablation (chaos_iters vs impls)\n",
           "_Note: vortex_fused is hard-coded at chaos_iters=5; only eager/sdpa-vortex rows vary._\n"]
    by_ci = {}
    for r in rows:
        by_ci.setdefault(r["cfg_chaos_iters"], {})[r["impl"]] = r
    impls = sorted({r["impl"] for r in rows})
    cis = sorted(by_ci.keys(), key=int)
    out.append("### fwd_ms")
    out.append("| chaos_iters | " + " | ".join(impls) + " |")
    out.append("|---|" + "|".join("---" for _ in impls) + "|")
    for ci in cis:
        cells = []
        for imp in impls:
            r = by_ci[ci].get(imp)
            cells.append(_fmt_float(r["fwd_ms"]) if (r and r.get("ok") == "1") else "x")
        out.append(f"| {ci} | " + " | ".join(cells) + " |")
    out.append("\n### bwd_ms")
    out.append("| chaos_iters | " + " | ".join(impls) + " |")
    out.append("|---|" + "|".join("---" for _ in impls) + "|")
    for ci in cis:
        cells = []
        for imp in impls:
            r = by_ci[ci].get(imp)
            cells.append(_fmt_float(r["bwd_ms"]) if (r and r.get("ok") == "1") else "x")
        out.append(f"| {ci} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel_cfg", type=str, default=None)
    ap.add_argument("--seqlen", type=str, default=None)
    ap.add_argument("--chaos", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    body = ["# Vortex-Helix fused kernel — autoresearch summary\n"]
    body.append(_section_kernel_cfg(_read_tsv(args.kernel_cfg)))
    body.append(_section_seqlen(_read_tsv(args.seqlen)))
    body.append(_section_chaos(_read_tsv(args.chaos)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(body))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
