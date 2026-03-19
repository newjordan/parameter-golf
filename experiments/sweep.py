#!/usr/bin/env python3
"""
Phase 1 Experiment Sweep Orchestrator

Runs parameter-golf experiments sequentially or in parallel, logging results
to experiments/results.jsonl. Supports both H100 (8-GPU distributed) and
DGX Spark / H10 (single GPU) configurations.

Usage:
    # Run a single experiment
    python experiments/sweep.py --experiment lr_matrix_0.05 --hardware h100

    # Run a full category sweep
    python experiments/sweep.py --category lr_matrix --hardware spark

    # Run all Phase 1 experiments
    python experiments/sweep.py --all --hardware h100

    # Resume from where you left off (skips completed experiments)
    python experiments/sweep.py --all --hardware h100 --resume

    # Dry run — print experiments without executing
    python experiments/sweep.py --all --hardware h100 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = REPO_ROOT / "train_gpt.py"
RESULTS_FILE = REPO_ROOT / "experiments" / "results.jsonl"
LOGS_DIR = REPO_ROOT / "experiments" / "logs"
BRANCH = "claude/optimize-train-baseline-M8rn2"

# ---------------------------------------------------------------------------
# Hardware presets
# ---------------------------------------------------------------------------

HARDWARE_PRESETS = {
    "h100": {
        "launch_cmd": ["torchrun", "--nproc_per_node=8"],
        "env_overrides": {
            "MAX_WALLCLOCK_SECONDS": "600",
        },
        "description": "8×H100 (official leaderboard config)",
    },
    "spark": {
        "launch_cmd": ["python"],
        "env_overrides": {
            "MAX_WALLCLOCK_SECONDS": "1800",
            "ITERATIONS": "50000",
            "VAL_LOSS_EVERY": "2000",
            "TRAIN_LOG_EVERY": "500",
        },
        "description": "DGX Spark / H10 (single GPU research proxy)",
    },
    "spark_quick": {
        "launch_cmd": ["python"],
        "env_overrides": {
            "MAX_WALLCLOCK_SECONDS": "600",
            "ITERATIONS": "20000",
            "VAL_LOSS_EVERY": "1000",
            "TRAIN_LOG_EVERY": "200",
        },
        "description": "DGX Spark / H10 quick run (10 min cap)",
    },
}

# ---------------------------------------------------------------------------
# Experiment definitions — Phase 1
# ---------------------------------------------------------------------------

def make_single_sweep(param: str, values: list, prefix: str | None = None) -> dict:
    """Generate a sweep over a single env var."""
    tag = prefix or param.lower()
    return {
        f"{tag}_{v}": {param: str(v)}
        for v in values
    }


# Category 1: Learning Rate Tuning
EXPERIMENTS_LR_MATRIX = make_single_sweep(
    "MATRIX_LR", [0.02, 0.03, 0.035, 0.045, 0.05, 0.06, 0.08], "lr_matrix"
)
EXPERIMENTS_LR_SCALAR = make_single_sweep(
    "SCALAR_LR", [0.02, 0.03, 0.05, 0.06, 0.08], "lr_scalar"
)
EXPERIMENTS_LR_EMBED = make_single_sweep(
    "TIED_EMBED_LR", [0.03, 0.04, 0.06, 0.08, 0.1], "lr_embed"
)

# Category 2: Batch Size
EXPERIMENTS_BATCH = make_single_sweep(
    "TRAIN_BATCH_TOKENS", [262144, 393216, 786432, 1048576], "batch"
)

# Category 3: Warmup & Warmdown Schedule
EXPERIMENTS_WARMDOWN = make_single_sweep(
    "WARMDOWN_ITERS", [800, 1000, 1500, 2000, 2500, 3000], "warmdown"
)
EXPERIMENTS_MOMENTUM = make_single_sweep(
    "MUON_MOMENTUM", [0.90, 0.93, 0.97], "momentum"
)
EXPERIMENTS_MOM_WARMUP_START = make_single_sweep(
    "MUON_MOMENTUM_WARMUP_START", [0.80, 0.90], "mom_warmup_start"
)
EXPERIMENTS_MOM_WARMUP_STEPS = make_single_sweep(
    "MUON_MOMENTUM_WARMUP_STEPS", [200, 800, 1000], "mom_warmup_steps"
)

# Category 4: Regularization & Gradient Control
EXPERIMENTS_GRAD_CLIP = make_single_sweep(
    "GRAD_CLIP_NORM", [0.5, 1.0, 2.0, 5.0], "grad_clip"
)
EXPERIMENTS_BETA1 = make_single_sweep("BETA1", [0.85, 0.95], "beta1")
EXPERIMENTS_BETA2 = make_single_sweep("BETA2", [0.90, 0.98, 0.99], "beta2")

# Category 5: Architecture-Adjacent Tweaks
EXPERIMENTS_QK_GAIN = make_single_sweep(
    "QK_GAIN_INIT", [1.0, 1.25, 1.75, 2.0], "qk_gain"
)
EXPERIMENTS_SOFTCAP = make_single_sweep(
    "LOGIT_SOFTCAP", [20.0, 25.0, 40.0, 50.0], "softcap"
)
EXPERIMENTS_ROPE = make_single_sweep(
    "ROPE_BASE", [5000, 50000, 100000], "rope"
)
EXPERIMENTS_SEQLEN = make_single_sweep(
    "TRAIN_SEQ_LEN", [512, 768, 1536, 2048], "seqlen"
)

# Category 6: Model Shape
EXPERIMENTS_SHAPE = {
    "shape_7x576": {"NUM_LAYERS": "7", "MODEL_DIM": "576"},
    "shape_11x448": {"NUM_LAYERS": "11", "MODEL_DIM": "448"},
    "shape_heads16_kv4": {"NUM_HEADS": "16", "NUM_KV_HEADS": "4"},
    "shape_heads8_kv2": {"NUM_HEADS": "8", "NUM_KV_HEADS": "2"},
    "shape_heads8_kv8": {"NUM_HEADS": "8", "NUM_KV_HEADS": "8"},
    "shape_mlp3_dim420": {"MLP_MULT": "3", "MODEL_DIM": "420"},
}

# Baseline (no changes, for comparison)
EXPERIMENTS_BASELINE = {
    "baseline": {},
}

# All categories in recommended execution order
CATEGORY_MAP = {
    "baseline": EXPERIMENTS_BASELINE,
    "lr_matrix": EXPERIMENTS_LR_MATRIX,
    "lr_scalar": EXPERIMENTS_LR_SCALAR,
    "lr_embed": EXPERIMENTS_LR_EMBED,
    "batch": EXPERIMENTS_BATCH,
    "warmdown": EXPERIMENTS_WARMDOWN,
    "momentum": EXPERIMENTS_MOMENTUM,
    "mom_warmup_start": EXPERIMENTS_MOM_WARMUP_START,
    "mom_warmup_steps": EXPERIMENTS_MOM_WARMUP_STEPS,
    "grad_clip": EXPERIMENTS_GRAD_CLIP,
    "beta1": EXPERIMENTS_BETA1,
    "beta2": EXPERIMENTS_BETA2,
    "qk_gain": EXPERIMENTS_QK_GAIN,
    "softcap": EXPERIMENTS_SOFTCAP,
    "rope": EXPERIMENTS_ROPE,
    "seqlen": EXPERIMENTS_SEQLEN,
    "shape": EXPERIMENTS_SHAPE,
}

ALL_EXPERIMENTS: dict[str, dict[str, str]] = {}
for cat_exps in CATEGORY_MAP.values():
    ALL_EXPERIMENTS.update(cat_exps)


# ---------------------------------------------------------------------------
# Git sync — push results back to the shared branch
# ---------------------------------------------------------------------------

def git_sync_pull() -> bool:
    """Pull latest results from remote before starting. Returns True on success."""
    try:
        # Fetch + merge results from other machines
        subprocess.run(
            ["git", "fetch", "origin", BRANCH],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        # Only merge the results file and logs to avoid conflicts with code changes
        subprocess.run(
            ["git", "merge", f"origin/{BRANCH}", "--no-edit", "-X", "ours"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
        print("[git-sync] Pulled latest results from remote.")
        return True
    except Exception as e:
        print(f"[git-sync] Pull failed (non-fatal): {e}")
        return False


def git_sync_push(experiment_id: str, hardware: str) -> bool:
    """Commit and push results after an experiment completes."""
    try:
        # Stage results file and the experiment log
        files_to_add = [str(RESULTS_FILE)]
        log_file = LOGS_DIR / f"{experiment_id}.log"
        if log_file.exists():
            files_to_add.append(str(log_file))

        subprocess.run(
            ["git", "add"] + files_to_add,
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )

        # Check if there's anything to commit
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(REPO_ROOT), capture_output=True, timeout=10,
        )
        if status.returncode == 0:
            # Nothing staged
            return True

        subprocess.run(
            ["git", "commit", "-m", f"results: {experiment_id} on {hardware}"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )

        # Push with retry (exponential backoff)
        for attempt, delay in enumerate([0, 2, 4, 8], 1):
            if delay > 0:
                time.sleep(delay)
            result = subprocess.run(
                ["git", "push", "-u", "origin", BRANCH],
                cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                print(f"[git-sync] Pushed {experiment_id} results (attempt {attempt}).")
                return True
            # If push fails due to remote changes, pull and retry
            if "rejected" in result.stderr or "fetch first" in result.stderr:
                subprocess.run(
                    ["git", "pull", "--rebase", "origin", BRANCH],
                    cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
                )

        print(f"[git-sync] Push failed after 4 attempts. Results saved locally.")
        return False

    except Exception as e:
        print(f"[git-sync] Sync failed (non-fatal): {e}")
        return False


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def load_completed(results_file: Path) -> set[str]:
    """Load experiment IDs that have already been completed."""
    completed = set()
    if results_file.exists():
        for line in results_file.read_text().strip().split("\n"):
            if line.strip():
                try:
                    rec = json.loads(line)
                    completed.add(rec["experiment_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def parse_training_log(log_text: str) -> dict:
    """Extract key metrics from training output."""
    result = {
        "val_bpb_prequant": None,
        "val_bpb_postquant": None,
        "artifact_bytes": None,
        "training_time_ms": None,
        "steps_completed": None,
    }

    # Find all val_bpb entries (last one before quantization is pre-quant)
    val_bpb_matches = re.findall(r"step:(\d+)/\d+ val_loss:[\d.]+ val_bpb:([\d.]+)", log_text)
    if val_bpb_matches:
        result["steps_completed"] = int(val_bpb_matches[-1][0])
        result["val_bpb_prequant"] = float(val_bpb_matches[-1][1])

    # Post-quant roundtrip
    roundtrip = re.search(r"final_int8_zlib_roundtrip val_loss:[\d.]+ val_bpb:([\d.]+)", log_text)
    if roundtrip:
        result["val_bpb_postquant"] = float(roundtrip.group(1))

    # Artifact size
    artifact = re.search(r"Total submission size int8\+zlib: (\d+) bytes", log_text)
    if artifact:
        result["artifact_bytes"] = int(artifact.group(1))

    # Training time
    train_time = re.search(r"train_time:(\d+)ms", log_text)
    if train_time:
        result["training_time_ms"] = int(train_time.group(1))

    # Also check for stopping early
    early_stop = re.search(r"stopping_early.*step:(\d+)/", log_text)
    if early_stop:
        result["steps_completed"] = int(early_stop.group(1))

    return result


def run_experiment(
    experiment_id: str,
    env_vars: dict[str, str],
    hardware: str,
    dry_run: bool = False,
    seed: int | None = None,
    sync: bool = True,
) -> dict | None:
    """Run a single experiment and return parsed results."""
    preset = HARDWARE_PRESETS[hardware]

    # Build environment
    env = os.environ.copy()
    env.update(preset["env_overrides"])
    env.update(env_vars)
    if seed is not None:
        env["SEED"] = str(seed)
    env["RUN_ID"] = experiment_id

    # Build command
    cmd = preset["launch_cmd"] + [str(TRAIN_SCRIPT)]

    if dry_run:
        env_str = " ".join(f"{k}={v}" for k, v in env_vars.items())
        print(f"  [DRY RUN] {experiment_id}: {env_str}")
        print(f"            cmd: {' '.join(cmd)}")
        return None

    # Ensure log directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"{experiment_id}.log"

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_id}")
    print(f"  Hardware: {preset['description']}")
    print(f"  Env vars: {env_vars}")
    print(f"  Command:  {' '.join(cmd)}")
    print(f"  Log file: {log_file}")
    print(f"{'='*70}")

    start_time = time.time()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour hard timeout
        )
        wall_time = time.time() - start_time

        # Save full log
        full_log = proc.stdout + "\n" + proc.stderr
        log_file.write_text(full_log)

        if proc.returncode != 0:
            print(f"  FAILED (exit code {proc.returncode})")
            print(f"  stderr (last 500 chars): {proc.stderr[-500:]}")
            return None

        # Parse results
        metrics = parse_training_log(full_log)
        result = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "env_vars": env_vars,
            "hardware": hardware,
            "wall_time_seconds": round(wall_time, 1),
            **metrics,
        }

        # Print summary
        bpb_pre = metrics.get("val_bpb_prequant", "N/A")
        bpb_post = metrics.get("val_bpb_postquant", "N/A")
        artifact = metrics.get("artifact_bytes", "N/A")
        print(f"  RESULT: pre-quant={bpb_pre} post-quant={bpb_post} artifact={artifact} bytes")

        # Append to results file
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Auto-sync results to git
        if sync:
            git_sync_push(experiment_id, hardware)

        return result

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {time.time() - start_time:.0f}s")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def print_leaderboard():
    """Print current best results from results.jsonl."""
    if not RESULTS_FILE.exists():
        print("No results yet.")
        return

    results = []
    for line in RESULTS_FILE.read_text().strip().split("\n"):
        if line.strip():
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not results:
        print("No valid results found.")
        return

    # Sort by post-quant BPB (lower is better)
    valid = [r for r in results if r.get("val_bpb_postquant") is not None]
    valid.sort(key=lambda r: r["val_bpb_postquant"])

    print(f"\n{'='*90}")
    print(f"LEADERBOARD ({len(valid)} experiments with valid post-quant BPB)")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Experiment':<30} {'Pre-Q BPB':<12} {'Post-Q BPB':<12} {'Artifact':<12} {'Hardware':<10}")
    print(f"{'-'*5} {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for i, r in enumerate(valid[:20], 1):
        name = r["experiment_id"][:30]
        pre = f"{r.get('val_bpb_prequant', 0):.4f}" if r.get("val_bpb_prequant") else "N/A"
        post = f"{r['val_bpb_postquant']:.4f}"
        art = f"{r.get('artifact_bytes', 0):,}" if r.get("artifact_bytes") else "N/A"
        hw = r.get("hardware", "?")
        print(f"{i:<5} {name:<30} {pre:<12} {post:<12} {art:<12} {hw:<10}")

    if valid:
        best = valid[0]
        print(f"\nBEST: {best['experiment_id']} → {best['val_bpb_postquant']:.4f} BPB")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Experiment Sweep")
    parser.add_argument("--hardware", choices=list(HARDWARE_PRESETS.keys()), required=True,
                        help="Hardware target")
    parser.add_argument("--experiment", type=str, help="Run a single named experiment")
    parser.add_argument("--category", type=str, choices=list(CATEGORY_MAP.keys()),
                        help="Run all experiments in a category")
    parser.add_argument("--all", action="store_true", help="Run all Phase 1 experiments")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiments without executing")
    parser.add_argument("--leaderboard", action="store_true",
                        help="Print current leaderboard and exit")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed for all experiments")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed validation (e.g. 1337,42,7)")
    parser.add_argument("--no-sync", action="store_true",
                        help="Disable auto git commit+push after each experiment")
    args = parser.parse_args()

    if args.leaderboard:
        print_leaderboard()
        return

    # Determine which experiments to run
    experiments: dict[str, dict[str, str]] = {}
    if args.experiment:
        if args.experiment in ALL_EXPERIMENTS:
            experiments[args.experiment] = ALL_EXPERIMENTS[args.experiment]
        else:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {', '.join(sorted(ALL_EXPERIMENTS.keys()))}")
            sys.exit(1)
    elif args.category:
        experiments = CATEGORY_MAP[args.category]
    elif args.all:
        experiments = ALL_EXPERIMENTS
    else:
        parser.print_help()
        sys.exit(1)

    # Filter completed if resuming
    if args.resume:
        completed = load_completed(RESULTS_FILE)
        before = len(experiments)
        experiments = {k: v for k, v in experiments.items() if k not in completed}
        skipped = before - len(experiments)
        if skipped > 0:
            print(f"Resuming: skipping {skipped} completed experiments")

    if not experiments:
        print("No experiments to run.")
        print_leaderboard()
        return

    # Pull latest results from remote before starting
    sync = not args.no_sync
    if sync and not args.dry_run:
        git_sync_pull()
        # Re-check completed after pull (other machine may have finished some)
        if args.resume:
            completed = load_completed(RESULTS_FILE)
            experiments = {k: v for k, v in experiments.items() if k not in completed}
            if not experiments:
                print("All experiments already completed (after pulling remote results).")
                print_leaderboard()
                return

    print(f"Queued {len(experiments)} experiments on {args.hardware}")
    if args.dry_run:
        print("DRY RUN mode — no experiments will be executed\n")

    # Handle multi-seed validation
    seeds = [args.seed] if args.seed else [None]
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]

    total = len(experiments) * len(seeds)
    completed_count = 0

    for exp_id, env_vars in experiments.items():
        for seed in seeds:
            completed_count += 1
            run_id = exp_id if seed is None else f"{exp_id}_seed{seed}"
            print(f"\n[{completed_count}/{total}] Running {run_id}...")
            run_experiment(run_id, env_vars, args.hardware, dry_run=args.dry_run, seed=seed, sync=sync)

    if not args.dry_run:
        print_leaderboard()


if __name__ == "__main__":
    main()
