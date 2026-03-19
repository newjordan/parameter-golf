#!/usr/bin/env python3
"""
Compare experiment rankings between H100 and Spark to validate transfer.

Usage:
    python experiments/compare_hardware.py
    python experiments/compare_hardware.py --category lr_matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_FILE = Path(__file__).resolve().parent / "results.jsonl"


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    results = []
    for line in RESULTS_FILE.read_text().strip().split("\n"):
        if line.strip():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def compare(category: str | None = None):
    results = load_results()
    if not results:
        print("No results found.")
        return

    # Split by hardware
    h100 = {}
    spark = {}
    for r in results:
        bpb = r.get("val_bpb_postquant")
        if bpb is None:
            continue
        # Strip seed suffix for matching
        exp_id = r["experiment_id"].split("_seed")[0]
        hw = r.get("hardware", "unknown")
        if hw == "h100":
            h100[exp_id] = bpb
        elif hw in ("spark", "spark_quick"):
            spark[exp_id] = bpb

    # Find common experiments
    common = set(h100.keys()) & set(spark.keys())
    if category:
        from sweep import CATEGORY_MAP
        if category in CATEGORY_MAP:
            cat_ids = set(CATEGORY_MAP[category].keys())
            common = common & cat_ids

    if not common:
        print(f"No common experiments found between H100 and Spark.")
        print(f"  H100 experiments: {len(h100)}")
        print(f"  Spark experiments: {len(spark)}")
        return

    # Rank both
    h100_ranked = sorted(common, key=lambda x: h100[x])
    spark_ranked = sorted(common, key=lambda x: spark[x])

    h100_rank = {exp: i + 1 for i, exp in enumerate(h100_ranked)}
    spark_rank = {exp: i + 1 for i, exp in enumerate(spark_ranked)}

    print(f"\n{'='*80}")
    print(f"Hardware Transfer Validation ({len(common)} common experiments)")
    print(f"{'='*80}")
    print(f"{'Experiment':<30} {'H100 BPB':<12} {'H100 Rank':<10} {'Spark BPB':<12} {'Spark Rank':<10} {'Rank Δ':<8}")
    print(f"{'-'*30} {'-'*12} {'-'*10} {'-'*12} {'-'*10} {'-'*8}")

    for exp in h100_ranked:
        h_bpb = f"{h100[exp]:.4f}"
        s_bpb = f"{spark[exp]:.4f}"
        h_rank = h100_rank[exp]
        s_rank = spark_rank[exp]
        delta = abs(h_rank - s_rank)
        flag = " !" if delta > len(common) // 3 else ""
        print(f"{exp:<30} {h_bpb:<12} {h_rank:<10} {s_bpb:<12} {s_rank:<10} {delta:<8}{flag}")

    # Spearman rank correlation (simplified)
    n = len(common)
    if n >= 3:
        d_sq_sum = sum((h100_rank[e] - spark_rank[e]) ** 2 for e in common)
        spearman = 1 - (6 * d_sq_sum) / (n * (n**2 - 1))
        print(f"\nSpearman rank correlation: {spearman:.3f}")
        if spearman > 0.8:
            print("Strong transfer — Spark rankings reliably predict H100 rankings.")
        elif spearman > 0.5:
            print("Moderate transfer — Use Spark for filtering, validate top picks on H100.")
        else:
            print("Weak transfer — Spark rankings diverge from H100. Prioritize H100 runs.")

    # Top picks
    print(f"\nH100 top 3: {', '.join(h100_ranked[:3])}")
    print(f"Spark top 3: {', '.join(spark_ranked[:3])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()
    compare(args.category)
