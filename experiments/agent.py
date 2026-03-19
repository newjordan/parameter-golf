#!/usr/bin/env python3
"""
Autonomous Experiment Agent — Grok-powered optimizer loop

Reads experiment results, asks Grok to analyze and propose the next experiment,
then runs it. Fully autonomous — runs unattended overnight.

Usage:
    # Set your xAI API key
    export XAI_API_KEY="xai-..."

    # Run on H100 (fully autonomous)
    python experiments/agent.py --hardware h100

    # Run on Spark (research proxy)
    python experiments/agent.py --hardware spark

    # Limit total experiments
    python experiments/agent.py --hardware h100 --max-experiments 100

    # Dry run (see what Grok proposes without executing)
    python experiments/agent.py --hardware h100 --dry-run

    # Use a different model
    python experiments/agent.py --hardware h100 --model grok-3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Import from sweep.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import (
    ALL_EXPERIMENTS,
    HARDWARE_PRESETS,
    LOGS_DIR,
    REPO_ROOT,
    RESULTS_FILE,
    git_sync_pull,
    git_sync_push,
    load_completed,
    print_leaderboard,
    run_experiment,
)

# ---------------------------------------------------------------------------
# Grok API client (OpenAI-compatible, no dependencies)
# ---------------------------------------------------------------------------

XAI_API_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4.20-0309-reasoning"


def call_grok(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 2000,
) -> str:
    """Call xAI Grok API using urllib (no openai dependency needed)."""
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "XAI_API_KEY not set. Export it: export XAI_API_KEY='xai-...'"
        )

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")

    # Bypass any container proxy settings (RunPod sets http_proxy which
    # urllib picks up but that proxy blocks external HTTPS)
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)

    req = urllib.request.Request(
        XAI_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    for attempt in range(4):
        try:
            with opener.open(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < 3:
                delay = 2 ** (attempt + 1)
                print(f"  [grok] API call failed ({e}), retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise RuntimeError(f"Grok API failed after 4 attempts: {e}") from e


# ---------------------------------------------------------------------------
# System prompt — tells Grok what it's optimizing and how
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert ML researcher optimizing a GPT language model for the Parameter Golf challenge.

OBJECTIVE: Minimize val_bpb (bits-per-byte) on the FineWeb validation set.
CONSTRAINTS:
- Compressed artifact (code + int8+zlib weights) must be ≤ 16,000,000 bytes
- Training must complete within the wallclock cap
- All changes are via environment variables (no code edits)

AVAILABLE ENVIRONMENT VARIABLES (with current defaults):
- MATRIX_LR=0.04 (Muon optimizer LR for weight matrices)
- SCALAR_LR=0.04 (Adam LR for vectors/scalars)
- TIED_EMBED_LR=0.05 (Adam LR for tied embedding)
- TRAIN_BATCH_TOKENS=524288 (tokens per step, across all GPUs)
- TRAIN_SEQ_LEN=1024 (sequence length)
- WARMDOWN_ITERS=1200 (LR decay window)
- MUON_MOMENTUM=0.95 (final Muon momentum)
- MUON_MOMENTUM_WARMUP_START=0.85 (initial momentum)
- MUON_MOMENTUM_WARMUP_STEPS=500 (momentum ramp steps)
- MUON_BACKEND_STEPS=5 (Newton-Schulz iterations)
- BETA1=0.9 (Adam beta1)
- BETA2=0.95 (Adam beta2)
- GRAD_CLIP_NORM=0.0 (gradient clipping, 0=disabled)
- QK_GAIN_INIT=1.5 (per-head Q gain init)
- LOGIT_SOFTCAP=30.0 (logit capping)
- ROPE_BASE=10000.0 (RoPE frequency base)
- NUM_LAYERS=9 (transformer layers)
- MODEL_DIM=512 (hidden dim)
- NUM_HEADS=8 (query heads)
- NUM_KV_HEADS=4 (key-value heads, GQA)
- MLP_MULT=2 (MLP hidden multiplier)
- TIE_EMBEDDINGS=1 (tie input/output embeddings)
- ITERATIONS=20000 (max training iterations)
- WARMUP_STEPS=20 (torch compile warmup)
- SEED=1337

RULES:
1. Propose ONE experiment at a time
2. Each experiment changes 1-3 env vars from the baseline (or from the current best)
3. Be bold but not reckless — try things that have theoretical motivation
4. Learn from failures — if increasing LR hurts, don't try even higher
5. Look for interactions — if higher LR + higher momentum both help individually, try them together
6. Track the artifact size — model shape changes affect the 16MB budget
7. After finding improvements, combine them into a "champion" config
8. Validate champion configs with multiple seeds

YOUR RESPONSE FORMAT (strict JSON, no markdown):
{
  "reasoning": "Brief explanation of why this experiment (2-3 sentences max)",
  "experiment_id": "descriptive_name_here",
  "env_vars": {"VAR_NAME": "value", ...},
  "category": "exploration|exploitation|combination|validation",
  "based_on": "experiment_id of the config this builds on, or 'baseline'"
}"""


# ---------------------------------------------------------------------------
# Build the analysis prompt from current results
# ---------------------------------------------------------------------------

def build_analysis_prompt(hardware: str, experiment_count: int) -> str:
    """Build a prompt with all results so far for Grok to analyze."""
    results = []
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().strip().split("\n"):
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Filter to this hardware
    hw_results = [r for r in results if r.get("hardware") == hardware]

    if not hw_results:
        return (
            f"No experiments have been run yet on {hardware}. "
            f"Start with the baseline (no env var changes), then begin exploring "
            f"learning rates (MATRIX_LR, SCALAR_LR, TIED_EMBED_LR) since they "
            f"typically have the highest impact. This is experiment #{experiment_count + 1}."
        )

    # Sort by post-quant BPB
    valid = [r for r in hw_results if r.get("val_bpb_postquant") is not None]
    valid.sort(key=lambda r: r["val_bpb_postquant"])
    failed = [r for r in hw_results if r.get("val_bpb_postquant") is None]

    # Build summary table
    lines = [f"RESULTS SO FAR ({len(valid)} successful, {len(failed)} failed) on {hardware}:"]
    lines.append(f"{'Rank':<4} {'Experiment':<35} {'Post-Q BPB':<12} {'Pre-Q BPB':<12} {'Artifact':<12} {'Env Vars'}")
    lines.append("-" * 120)

    for i, r in enumerate(valid, 1):
        name = r["experiment_id"][:35]
        post = f"{r['val_bpb_postquant']:.4f}"
        pre = f"{r.get('val_bpb_prequant', 0):.4f}" if r.get("val_bpb_prequant") else "N/A"
        art = f"{r.get('artifact_bytes', 0):,}" if r.get("artifact_bytes") else "N/A"
        env = json.dumps(r.get("env_vars", {}))
        lines.append(f"{i:<4} {name:<35} {post:<12} {pre:<12} {art:<12} {env}")

    if failed:
        lines.append(f"\nFAILED experiments: {', '.join(r['experiment_id'] for r in failed)}")

    # Best config
    if valid:
        best = valid[0]
        lines.append(f"\nCURRENT BEST: {best['experiment_id']} → {best['val_bpb_postquant']:.4f} BPB")
        lines.append(f"  Config: {json.dumps(best.get('env_vars', {}))}")
        baseline_results = [r for r in valid if r["experiment_id"] == "baseline"]
        if baseline_results:
            bl = baseline_results[0]["val_bpb_postquant"]
            improvement = bl - best["val_bpb_postquant"]
            lines.append(f"  Improvement over baseline: {improvement:.4f} BPB ({improvement/bl*100:.2f}%)")

    lines.append(f"\nThis will be experiment #{experiment_count + 1}. Propose the next experiment.")

    # Add phase guidance based on experiment count
    if experiment_count < 5:
        lines.append("\nPHASE: Early exploration. Focus on single-variable sweeps of learning rates.")
    elif experiment_count < 20:
        lines.append("\nPHASE: Broad exploration. Sweep remaining categories (batch size, schedule, regularization).")
    elif experiment_count < 50:
        lines.append("\nPHASE: Exploitation. Combine the best settings. Try 2-3 var combinations.")
    elif experiment_count < 80:
        lines.append("\nPHASE: Fine-tuning. Narrow sweeps around the champion config.")
    else:
        lines.append("\nPHASE: Validation. Run champion config with seeds 42, 7, 2024 to confirm stability.")

    return "\n".join(lines)


def parse_grok_response(response_text: str) -> dict | None:
    """Parse Grok's JSON response, handling markdown code fences."""
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        # Validate required fields
        if "experiment_id" not in data or "env_vars" not in data:
            print(f"  [grok] Missing required fields in response: {data.keys()}")
            return None
        # Ensure env_vars values are strings
        data["env_vars"] = {k: str(v) for k, v in data["env_vars"].items()}
        return data
    except json.JSONDecodeError as e:
        print(f"  [grok] Failed to parse JSON: {e}")
        print(f"  [grok] Raw response: {text[:500]}")
        return None


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def agent_loop(
    hardware: str,
    model: str,
    max_experiments: int,
    dry_run: bool,
    temperature: float,
):
    """Run the autonomous experiment agent."""
    print(f"\n{'='*70}")
    print(f"  AUTONOMOUS AGENT — Phase 1 Parameter Golf")
    print(f"{'='*70}")
    print(f"  Hardware:    {HARDWARE_PRESETS[hardware]['description']}")
    print(f"  LLM:         {model} via xAI API")
    print(f"  Max runs:    {max_experiments}")
    print(f"  Temperature: {temperature}")
    print(f"  Dry run:     {dry_run}")
    print(f"  Results:     {RESULTS_FILE}")
    print(f"{'='*70}\n")

    # Pull latest before starting
    if not dry_run:
        git_sync_pull()

    agent_log = LOGS_DIR / "agent_decisions.jsonl"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    completed = load_completed(RESULTS_FILE)
    experiment_count = len(completed)

    consecutive_failures = 0
    max_consecutive_failures = 5

    for run_num in range(1, max_experiments + 1):
        print(f"\n{'─'*70}")
        print(f"AGENT STEP {run_num}/{max_experiments} (total experiments so far: {experiment_count})")
        print(f"{'─'*70}")

        # Build prompt with all results
        analysis = build_analysis_prompt(hardware, experiment_count)

        # Ask Grok
        print("  [grok] Analyzing results and proposing next experiment...")
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": analysis},
            ]
            response = call_grok(messages, model=model, temperature=temperature)
        except RuntimeError as e:
            print(f"  [grok] ERROR: {e}")
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"  Stopping: {max_consecutive_failures} consecutive failures.")
                break
            continue

        # Parse response
        proposal = parse_grok_response(response)
        if proposal is None:
            print("  [grok] Could not parse response. Retrying with lower temperature...")
            try:
                response = call_grok(messages, model=model, temperature=0.3)
                proposal = parse_grok_response(response)
            except RuntimeError:
                pass

        if proposal is None:
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                print(f"  Stopping: {max_consecutive_failures} consecutive parse failures.")
                break
            continue

        consecutive_failures = 0
        exp_id = proposal["experiment_id"]
        env_vars = proposal["env_vars"]
        reasoning = proposal.get("reasoning", "")
        category = proposal.get("category", "unknown")

        print(f"  [grok] Proposal: {exp_id}")
        print(f"  [grok] Category: {category}")
        print(f"  [grok] Reasoning: {reasoning}")
        print(f"  [grok] Env vars: {env_vars}")

        # Log the decision
        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_num": run_num,
            "experiment_id": exp_id,
            "env_vars": env_vars,
            "reasoning": reasoning,
            "category": category,
            "based_on": proposal.get("based_on", "unknown"),
            "grok_model": model,
            "hardware": hardware,
        }
        with open(agent_log, "a") as f:
            f.write(json.dumps(decision) + "\n")

        # Check for duplicates
        if exp_id in completed:
            print(f"  [skip] {exp_id} already completed. Telling Grok to try something else.")
            # Modify the analysis to tell Grok this was already done
            continue

        # Run it
        if dry_run:
            print(f"  [DRY RUN] Would run: {exp_id} with {env_vars}")
        else:
            result = run_experiment(
                experiment_id=exp_id,
                env_vars=env_vars,
                hardware=hardware,
                dry_run=False,
                sync=True,
            )

            if result is not None:
                experiment_count += 1
                completed.add(exp_id)
                bpb = result.get("val_bpb_postquant", "N/A")
                print(f"  [result] {exp_id} → post-quant BPB: {bpb}")
            else:
                print(f"  [result] {exp_id} → FAILED")

        # Brief pause between experiments (let GPU cool, avoid API rate limits)
        if not dry_run and run_num < max_experiments:
            time.sleep(5)

    print("\n" + "=" * 70)
    print("AGENT COMPLETE")
    print("=" * 70)
    print_leaderboard()

    # Push the agent decision log
    if not dry_run and agent_log.exists():
        try:
            import subprocess
            subprocess.run(
                ["git", "add", str(agent_log)],
                cwd=str(REPO_ROOT), capture_output=True, timeout=10,
            )
            subprocess.run(
                ["git", "commit", "-m", f"agent: {experiment_count} experiments on {hardware}"],
                cwd=str(REPO_ROOT), capture_output=True, timeout=10,
            )
            subprocess.run(
                ["git", "push", "-u", "origin", "claude/optimize-train-baseline-M8rn2"],
                cwd=str(REPO_ROOT), capture_output=True, timeout=30,
            )
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Autonomous Grok-powered experiment agent")
    parser.add_argument("--hardware", choices=list(HARDWARE_PRESETS.keys()), required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Grok model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-experiments", type=int, default=100,
                        help="Maximum experiments to run (default: 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show proposals without running experiments")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Grok sampling temperature (default: 0.7, higher=more chaotic)")
    args = parser.parse_args()

    agent_loop(
        hardware=args.hardware,
        model=args.model,
        max_experiments=args.max_experiments,
        dry_run=args.dry_run,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
