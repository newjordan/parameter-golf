"""Summarise autores_t_depth TSV into a T x depth speedup grid vs selected
baselines. Invoke as:
    python -m bench.summarize_t_depth records/autores_t_depth_v2.tsv
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "records/autores_t_depth_v2.tsv"

by_key = defaultdict(dict)
with open(path) as f:
    rd = csv.DictReader(f, delimiter="\t")
    for row in rd:
        try:
            T = int(row["T"])
            d = int(row["cfg_chaos_iters"])
        except (ValueError, KeyError):
            continue
        key = (T, d)
        fwd = row["fwd_p50_ms"]
        bwd = row["bwd_p50_ms"]
        if fwd == "nan" or bwd == "nan":
            continue
        by_key[key][row["impl"]] = (float(fwd), float(bwd))

ts = sorted({k[0] for k in by_key})
ds = sorted({k[1] for k in by_key})
baselines = ["eager_vortex", "sdpa_flash_vortex", "sdpa_flash_attn_only"]
for bl in baselines:
    print(f"\n### speedup (fwd / bwd) vs {bl}")
    head = "T \\ d | " + " | ".join(str(d) for d in ds)
    print(head)
    print("---|" + "|".join("---" for _ in ds))
    for T in ts:
        cells = []
        for d in ds:
            vf = by_key.get((T, d), {}).get("vortex_fused")
            bb = by_key.get((T, d), {}).get(bl)
            if vf is None or bb is None:
                cells.append("-")
                continue
            fwd_sp = bb[0] / vf[0]
            bwd_sp = bb[1] / vf[1]
            cells.append(f"{fwd_sp:.2f}x / {bwd_sp:.2f}x")
        print(f"{T:5d} | " + " | ".join(cells))

print("\n### raw vortex_fused latency (fwd / bwd ms, p50)")
print("T \\ d | " + " | ".join(str(d) for d in ds))
print("---|" + "|".join("---" for _ in ds))
for T in ts:
    row = []
    for d in ds:
        vf = by_key.get((T, d), {}).get("vortex_fused")
        row.append("-" if vf is None else f"{vf[0]:.3f} / {vf[1]:.3f}")
    print(f"{T:5d} | " + " | ".join(row))
