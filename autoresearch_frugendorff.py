"""
Frugendorff Auto-Research: Qwen-Guided TTT + Fractal Calibration
=================================================================
Exhaustively calibrate the Frugendorff's TTT, cadence, loop count,
drift gate, LR, and architecture to find its best use case.

Tests on DGX Spark with train_fractal_cadence.py.
Qwen analyzes results and proposes next experiment.

Usage:
  source .venv/bin/activate
  nohup python autoresearch_frugendorff.py > autoresearch_frug.log 2>&1 &
"""

import csv
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

SCRIPT = "train_fractal_cadence.py"
RESULTS_FILE = "autoresearch_frug_results.csv"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-coder:30b")

FIELDS = [
    "timestamp", "run_id", "val_bpb",
    "cadence", "cadence_offset", "num_unique_layers", "num_loops",
    "lr", "grad_clip", "mlp_mult", "model_dim",
    "steps", "f_steps", "n_steps", "avg_ms", "time_s", "params",
    "reasoning", "notes"
]

RUN_DEFAULTS = {
    "iterations": 500,
    "eval_tokens": 100000,
    "max_seconds": 600,
    "batch_tokens": 32768,
    "seq_len": 1024,
    "seed": 1337,
}

SYSTEM_PROMPT = """You are calibrating the "Frugendorff" — a fractal weight-shared transformer with cadence training.

GOAL: Find the absolute best configuration for this architecture. Minimize val_bpb.

ARCHITECTURE:
- Weight-shared blocks: N unique transformer blocks looped L times = N*L effective depth
- Cadence training: alternates fractal steps (all loops, deep) and normalize steps (single pass, fast)
  - cadence=1: always fractal. cadence=2: F/N/F/N. cadence=3: F/N/N. cadence=0: never fractal.
- Orthogonal loop positions: QR-initialized, each loop + normalize gets its own subspace
- U-Net skip connections within each loop iteration

KEY FINDINGS SO FAR:
- On H100: 3 blocks x 4 loops, dim=960, cadence=3 hit 1.2113 BPB (sliding window)
- With TTT at window 1400: peaked at 1.1901 then drifted back up
- Overnight Qwen sweep (141 runs) found: 2L x 4 loops, lr=2e-3, clip=5.0, mlp=3 was best locally
- Bigger batch (1.5x) was a wash — fewer steps offset richer gradients
- MLP 3.3 vs 3.0 was marginal
- The architecture works as compression: fewer unique params, same effective depth

WHAT WE NEED TO UNDERSTAND:
1. What's the optimal loops-to-layers ratio? (2x4, 3x3, 3x4, 4x3, 6x2, etc.)
2. Does cadence actually help or is always-fractal better with enough steps?
3. What LR / grad_clip combo works best for each configuration?
4. Is there a dim sweet spot? (wider = more expressive but fewer steps)
5. Does MLP mult interact with loop count? (4x MLP + fewer loops vs 3x MLP + more loops)
6. At what point does adding loops have diminishing returns?
7. Is the Frugendorff best as a standalone arch or as a compression shim inside a larger model?

CONFIGURABLE PARAMETERS:
- num_unique_layers: 1-8
- num_loops: 1-6
- cadence: 0-5 (0=never fractal, 1=always)
- cadence_offset: 0 to cadence-1
- lr: 1e-4 to 3e-3
- grad_clip: 0.3 to 10.0
- mlp_mult: 2, 3, 4
- model_dim: 0 (auto-size to match ~17M baseline) or specific value

LOCAL TEST: DGX Spark, 1 GPU, AdamW, 500 steps, ~3-5 min per run.
Relative rankings transfer to H100 even though absolute BPB doesn't.

BE SYSTEMATIC. Vary ONE axis at a time when exploiting. Vary MULTIPLE when exploring.
Don't repeat configs already tested. Track which axes have the most impact.

Respond with ONLY a JSON object:
{
  "reasoning": "2-3 sentences explaining the hypothesis",
  "config": {
    "num_unique_layers": <int>,
    "num_loops": <int>,
    "cadence": <int>,
    "cadence_offset": <int>,
    "lr": <float>,
    "grad_clip": <float>,
    "mlp_mult": <int>
  }
}"""

# ─── OLLAMA ───────────────────────────────────────────────────────────────────

def ask_qwen(history_text, last_result_text):
    prompt = f"""All results so far (sorted best first):

{history_text}

Most recent:
{last_result_text}

Propose the NEXT experiment. Be systematic — identify the most impactful axis and test it."""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 512}
    }
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data.get("message", {}).get("content", "")
    except Exception as e:
        print(f"  Qwen error: {e}")
        return None


def parse_response(text):
    if not text:
        return None, "no response"
    clean = text.strip()
    if "```" in clean:
        for p in clean.split("```"):
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                clean = p
                break
    start = clean.find("{")
    end = clean.rfind("}") + 1
    if start < 0 or end <= start:
        return None, f"no JSON: {text[:100]}"
    try:
        obj = json.loads(clean[start:end])
        reasoning = obj.get("reasoning", "")
        cfg = obj.get("config", obj)
        v = {}
        if "num_unique_layers" in cfg:
            v["num_unique_layers"] = max(1, min(8, int(cfg["num_unique_layers"])))
        if "num_loops" in cfg:
            v["num_loops"] = max(1, min(6, int(cfg["num_loops"])))
        if "cadence" in cfg:
            v["cadence"] = max(0, min(6, int(cfg["cadence"])))
        if "cadence_offset" in cfg:
            cad = v.get("cadence", 2)
            v["cadence_offset"] = max(0, min(max(cad - 1, 0), int(cfg["cadence_offset"])))
        if "lr" in cfg:
            v["lr"] = max(1e-5, min(0.01, float(cfg["lr"])))
        if "grad_clip" in cfg:
            v["grad_clip"] = max(0.1, min(10.0, float(cfg["grad_clip"])))
        if "mlp_mult" in cfg:
            v["mlp_mult"] = int(cfg["mlp_mult"])
            if v["mlp_mult"] not in [2, 3, 4]:
                v["mlp_mult"] = 2
        return v, reasoning
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return None, f"parse error: {e}"


# ─── RUNNER ───────────────────────────────────────────────────────────────────

def run_experiment(config, run_id):
    cfg = {**RUN_DEFAULTS, **config}
    cfg.setdefault("cadence", 2)
    cfg.setdefault("cadence_offset", 0)
    cfg.setdefault("num_unique_layers", 3)
    cfg.setdefault("num_loops", 3)
    cfg.setdefault("lr", 3e-4)
    cfg.setdefault("grad_clip", 1.0)
    cfg.setdefault("mlp_mult", 2)

    cmd = [
        sys.executable, SCRIPT,
        "--cadence", str(cfg["cadence"]),
        "--cadence-offset", str(cfg["cadence_offset"]),
        "--num-unique-layers", str(cfg["num_unique_layers"]),
        "--num-loops", str(cfg["num_loops"]),
        "--lr", str(cfg["lr"]),
        "--grad-clip", str(cfg["grad_clip"]),
        "--mlp-mult", str(cfg["mlp_mult"]),
        "--iterations", str(cfg["iterations"]),
        "--eval-tokens", str(cfg["eval_tokens"]),
        "--max-seconds", str(cfg["max_seconds"]),
        "--batch-tokens", str(cfg["batch_tokens"]),
        "--seq-len", str(cfg["seq_len"]),
        "--seed", str(cfg["seed"]),
        "--run-id", run_id,
    ]
    if cfg.get("model_dim", 0) > 0:
        cmd.extend(["--model-dim", str(cfg["model_dim"])])
    if cfg.get("gravity", False):
        cmd.append("--gravity")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        if result.stderr:
            print(f"  {result.stderr[-300:]}")
        return None

    parsed = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "cadence": cfg["cadence"], "cadence_offset": cfg["cadence_offset"],
        "num_unique_layers": cfg["num_unique_layers"], "num_loops": cfg["num_loops"],
        "lr": cfg["lr"], "grad_clip": cfg["grad_clip"],
        "mlp_mult": cfg["mlp_mult"], "model_dim": cfg.get("model_dim", 0),
    }
    stdout = result.stdout
    for line in stdout.split("\n"):
        if "val_bpb:" in line and "RESULTS" not in line and "val_bpb:enabled" not in line:
            try:
                for p in line.split():
                    if p.startswith("val_bpb:"):
                        parsed["val_bpb"] = float(p.split(":")[1])
            except (ValueError, IndexError):
                pass
        if line.startswith("steps:"):
            try:
                parts = line.split()
                parsed["steps"] = int(parts[0].split(":")[1])
                for p in parts:
                    if p.startswith("(F:"):
                        parsed["f_steps"] = int(p.split(":")[1])
                    if p.startswith("N:"):
                        parsed["n_steps"] = int(p.rstrip(")").split(":")[1])
            except (ValueError, IndexError):
                pass
        if "avg_ms:" in line:
            try:
                for p in line.split():
                    if p.startswith("avg_ms:"):
                        parsed["avg_ms"] = float(p.split(":")[1].rstrip("ms/step"))
            except (ValueError, IndexError):
                pass
        if "time:" in line and "train_time" not in line:
            try:
                for p in line.split():
                    if p.startswith("time:"):
                        parsed["time_s"] = float(p.split(":")[1].rstrip("s"))
            except (ValueError, IndexError):
                pass
        if "params:" in line and "model_params" not in line:
            try:
                for p in line.split():
                    if p.startswith("params:"):
                        parsed["params"] = p.split(":")[1].replace(",", "")
            except (ValueError, IndexError):
                pass
    return parsed


def format_history(results):
    if not results:
        return "No experiments yet."
    valid = [r for r in results if r.get("val_bpb") and float(r.get("val_bpb", 999)) < 100]
    valid.sort(key=lambda r: float(r["val_bpb"]))
    lines = []
    for r in valid[:40]:
        lines.append(
            f"bpb={float(r['val_bpb']):.4f} | "
            f"L={r.get('num_unique_layers','?')}x{r.get('num_loops','?')} "
            f"cad={r.get('cadence','?')} lr={float(r.get('lr',0)):.1e} "
            f"clip={float(r.get('grad_clip',0)):.1f} mlp={r.get('mlp_mult','?')} "
            f"| {r.get('notes','')[:40]}"
        )
    return "\n".join(lines)


def format_last(result):
    if not result:
        return "First run."
    return (
        f"bpb={result.get('val_bpb','?')} | L={result.get('num_unique_layers','?')}"
        f"x{result.get('num_loops','?')} cad={result.get('cadence','?')} "
        f"lr={result.get('lr','?')} clip={result.get('grad_clip','?')}"
    )


def load_results():
    results = []
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE) as f:
            for row in csv.DictReader(f):
                results.append(row)
    return results


def save_result(result):
    exists = Path(RESULTS_FILE).exists()
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(result)


def fallback_config():
    return {
        "num_unique_layers": random.choice([1, 2, 3, 4, 5, 6]),
        "num_loops": random.choice([1, 2, 3, 4, 5]),
        "cadence": random.choice([0, 1, 2, 3]),
        "cadence_offset": 0,
        "lr": random.choice([5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3]),
        "grad_clip": random.choice([1.0, 2.0, 5.0, 8.0]),
        "mlp_mult": random.choice([2, 3, 4]),
    }


# ─── SEEDS: systematic coverage ──────────────────────────────────────────────

SEEDS = [
    # Ratio exploration: loops vs layers
    {"num_unique_layers": 1, "num_loops": 6, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 3,
     "notes": "extreme: 1L x 6loops"},
    {"num_unique_layers": 2, "num_loops": 4, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 3,
     "notes": "prev best: 2x4"},
    {"num_unique_layers": 3, "num_loops": 3, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 3,
     "notes": "balanced: 3x3"},
    {"num_unique_layers": 4, "num_loops": 2, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 3,
     "notes": "wide: 4x2"},
    {"num_unique_layers": 6, "num_loops": 2, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 3,
     "notes": "hybrid-like: 6x2"},
    {"num_unique_layers": 6, "num_loops": 1, "cadence": 1, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 3,
     "notes": "no fractal: 6x1 (baseline)"},

    # Cadence sweep on best ratio
    {"num_unique_layers": 2, "num_loops": 4, "cadence": 1, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 3,
     "notes": "2x4 always fractal"},
    {"num_unique_layers": 2, "num_loops": 4, "cadence": 2, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 3,
     "notes": "2x4 cadence 2"},
    {"num_unique_layers": 2, "num_loops": 4, "cadence": 4, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 3,
     "notes": "2x4 cadence 4"},

    # MLP sweep
    {"num_unique_layers": 2, "num_loops": 4, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 2,
     "notes": "2x4 mlp 2"},
    {"num_unique_layers": 2, "num_loops": 4, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 4,
     "notes": "2x4 mlp 4"},

    # Compression use case: many unique layers, minimal loops
    {"num_unique_layers": 8, "num_loops": 2, "cadence": 3, "lr": 1e-3, "grad_clip": 5.0, "mlp_mult": 3,
     "notes": "compression: 8x2"},
    {"num_unique_layers": 5, "num_loops": 3, "cadence": 3, "lr": 2e-3, "grad_clip": 5.0, "mlp_mult": 3,
     "notes": "mid: 5x3"},
]

# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("FRUGENDORFF AUTO-RESEARCH — Deep Calibration")
    print(f"Model: {OLLAMA_MODEL} @ {OLLAMA_URL}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Results: {RESULTS_FILE}")
    print("=" * 70)

    results = load_results()
    run_count = len(results)
    last_result = None

    # Seed runs
    if run_count < len(SEEDS):
        print(f"\n>>> SEED PHASE: {len(SEEDS)} systematic configs")
        for i, cfg in enumerate(SEEDS):
            if i < run_count:
                continue
            run_count += 1
            rid = f"frug_{run_count:03d}"
            notes = cfg.pop("notes", "")
            print(f"\n[seed {run_count}] {notes}")
            print(f"  L={cfg.get('num_unique_layers')}x{cfg.get('num_loops')} "
                  f"cad={cfg.get('cadence')} mlp={cfg.get('mlp_mult')}")
            r = run_experiment(cfg, rid)
            if r:
                r["notes"] = notes
                r["reasoning"] = "seed"
                save_result(r)
                results.append(r)
                last_result = r
                print(f"  >>> val_bpb={r.get('val_bpb', '?')}")

    # Qwen-guided loop
    while True:
        run_count += 1
        best = min((float(r.get("val_bpb", 999)) for r in results if r.get("val_bpb")), default=999)
        print(f"\n{'='*70}")
        print(f"RUN {run_count} | {datetime.now().strftime('%H:%M:%S')} | best={best:.4f}")
        print(f"{'='*70}")

        response = ask_qwen(format_history(results), format_last(last_result))
        config = None
        reasoning = ""
        if response:
            config, reasoning = parse_response(response)
            if config:
                print(f"  Qwen: {reasoning[:120]}")
            else:
                print(f"  Parse fail: {reasoning[:100]}")

        if config is None:
            config = fallback_config()
            reasoning = "fallback random"

        cad = config.get("cadence", 2)
        if cad > 0:
            config["cadence_offset"] = min(config.get("cadence_offset", 0), max(cad - 1, 0))
        else:
            config["cadence_offset"] = 0

        print(f"  Config: L={config.get('num_unique_layers','?')}x{config.get('num_loops','?')} "
              f"cad={config.get('cadence','?')} lr={config.get('lr',0):.1e} "
              f"clip={config.get('grad_clip',0):.1f} mlp={config.get('mlp_mult','?')}")

        r = run_experiment(config, f"frug_{run_count:03d}")
        if r:
            r["reasoning"] = reasoning[:200]
            r["notes"] = reasoning[:100]
            save_result(r)
            results.append(r)
            last_result = r
            print(f"  >>> val_bpb={r.get('val_bpb', '?')}")

        if run_count % 5 == 0:
            valid = [r for r in results if r.get("val_bpb") and float(r.get("val_bpb", 999)) < 100]
            valid.sort(key=lambda r: float(r["val_bpb"]))
            print(f"\n{'='*80}")
            print(f"LEADERBOARD (top 15 of {len(valid)})")
            print(f"{'='*80}")
            for i, r in enumerate(valid[:15]):
                print(f"  {i+1:>2}. bpb={float(r['val_bpb']):>7.4f} | "
                      f"L={r.get('num_unique_layers','?')}x{r.get('num_loops','?')} "
                      f"cad={r.get('cadence','?')} lr={float(r.get('lr',0)):.1e} "
                      f"clip={float(r.get('grad_clip',0)):.1f} mlp={r.get('mlp_mult','?')}")


if __name__ == "__main__":
    main()
