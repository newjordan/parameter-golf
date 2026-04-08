#!/usr/bin/env python3
"""Guard crawler leg edits against overly broad train_gpt.py changes.

Usage:
  python3 scripts/leg_diff_guard.py legs/2026-04-07_BW22_LoopDepth_9F
  python3 scripts/leg_diff_guard.py legs/2026-04-07_BW22_LoopDepth_9F --write-lock
  python3 scripts/leg_diff_guard.py legs/2026-04-07_BW22_LoopDepth_9F --check-lock
  python3 scripts/leg_diff_guard.py path/to/train_gpt.py --parent legs/2026-03-29_BW5/train_gpt.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from pathlib import Path

PARENT_RE = re.compile(r"^Parent:\s*(\S+)\s*$")
LEG_RE = re.compile(r"^Leg:\s*(\S+)\s*$")
ENV_RE = re.compile(r'os\.environ\.get\("([A-Z0-9_]+)"')
LOCK_FILE_NAME = ".train_gpt.lock.json"


@dataclass
class DiffSummary:
    parent_path: Path
    target_path: Path
    total_changed_lines: int
    hyperparam_changed_lines: int
    hyperparam_envs: list[str]
    hyperparam_non_env_lines: int
    non_hyperparam_changed_lines: int
    hunks: int
    diff_text: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def relpath(path: Path) -> str:
    root = repo_root()
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def find_parent_from_leg_dir(leg_dir: Path) -> Path | None:
    for name in ("hypothesis.md", "ablation.md", "RESULTS.md"):
        candidate = leg_dir / name
        if not candidate.exists():
            continue
        for line in read_text(candidate).splitlines():
            m = PARENT_RE.match(line.strip())
            if not m:
                continue
            raw = m.group(1).strip()
            parent = (repo_root() / raw).resolve()
            return parent
    return None


def find_leader_parent() -> Path | None:
    leader = repo_root() / "LEADER.md"
    if not leader.exists():
        return None
    for line in read_text(leader).splitlines():
        m = LEG_RE.match(line.strip())
        if m:
            return (repo_root() / m.group(1).strip()).resolve()
    return None


def resolve_target_and_parent(target_arg: str, parent_arg: str | None) -> tuple[Path, Path]:
    target = Path(target_arg).resolve()
    if not target.exists():
        raise SystemExit(f"target does not exist: {target}")

    if target.is_dir():
        leg_dir = target
        target_file = leg_dir / "train_gpt.py"
    else:
        target_file = target
        leg_dir = target.parent

    if not target_file.exists():
        raise SystemExit(f"target train_gpt.py not found: {target_file}")

    if parent_arg:
        parent = Path(parent_arg).resolve()
    else:
        parent = find_parent_from_leg_dir(leg_dir) or find_leader_parent()
        if parent is None:
            raise SystemExit("could not infer parent; pass --parent explicitly")

    if parent.is_dir():
        parent_file = parent / "train_gpt.py"
    else:
        parent_file = parent

    if not parent_file.exists():
        raise SystemExit(f"parent train_gpt.py not found: {parent_file}")

    return target_file, parent_file


def find_hyperparameters_region(lines: list[str]) -> tuple[int, int]:
    start = None
    end = len(lines)
    for idx, line in enumerate(lines):
        if line.startswith("class Hyperparameters:"):
            start = idx
            continue
        if start is not None and idx > start and line.startswith("def "):
            end = idx
            break
    if start is None:
        return (-1, -1)
    return (start, end)


def is_nonblank_change(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return False
    return True


def analyze_diff(parent_path: Path, target_path: Path) -> DiffSummary:
    parent_text = read_text(parent_path)
    target_text = read_text(target_path)
    parent_lines = parent_text.splitlines()
    target_lines = target_text.splitlines()
    parent_hs, parent_he = find_hyperparameters_region(parent_lines)
    target_hs, target_he = find_hyperparameters_region(target_lines)

    if parent_hs < 0 or target_hs < 0:
        raise SystemExit("could not locate class Hyperparameters in one of the files")

    sm = SequenceMatcher(a=parent_lines, b=target_lines)
    total_changed_lines = 0
    hyperparam_changed_lines = 0
    non_hyperparam_changed_lines = 0
    hyperparam_non_env_lines = 0
    hyperparam_envs: set[str] = set()
    hunks = 0

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        hunks += 1
        removed = parent_lines[i1:i2]
        added = target_lines[j1:j2]
        changed = removed + added
        total_changed_lines += sum(1 for line in changed if is_nonblank_change(line))

        in_parent_hparams = parent_hs < i2 and i1 < parent_he
        in_target_hparams = target_hs < j2 and j1 < target_he
        if in_parent_hparams or in_target_hparams:
            for line in changed:
                if not is_nonblank_change(line):
                    continue
                hyperparam_changed_lines += 1
                envs = ENV_RE.findall(line)
                if envs:
                    hyperparam_envs.update(envs)
                else:
                    hyperparam_non_env_lines += 1
        else:
            non_hyperparam_changed_lines += sum(1 for line in changed if is_nonblank_change(line))

    diff_text = "".join(
        unified_diff(
            parent_lines,
            target_lines,
            fromfile=relpath(parent_path),
            tofile=relpath(target_path),
            lineterm="\n",
            n=3,
        )
    )

    return DiffSummary(
        parent_path=parent_path,
        target_path=target_path,
        total_changed_lines=total_changed_lines,
        hyperparam_changed_lines=hyperparam_changed_lines,
        hyperparam_envs=sorted(hyperparam_envs),
        hyperparam_non_env_lines=hyperparam_non_env_lines,
        non_hyperparam_changed_lines=non_hyperparam_changed_lines,
        hunks=hunks,
        diff_text=diff_text,
    )


def default_lock_path(target_path: Path) -> Path:
    return target_path.parent / LOCK_FILE_NAME


def write_lock(lock_path: Path, summary: DiffSummary) -> None:
    parent_text = read_text(summary.parent_path)
    target_text = read_text(summary.target_path)
    payload = {
        "target_path": relpath(summary.target_path),
        "parent_path": relpath(summary.parent_path),
        "target_sha256": sha256_text(target_text),
        "parent_sha256": sha256_text(parent_text),
        "summary": {
            "total_changed_lines": summary.total_changed_lines,
            "hyperparam_changed_lines": summary.hyperparam_changed_lines,
            "hyperparam_envs": summary.hyperparam_envs,
            "hyperparam_non_env_lines": summary.hyperparam_non_env_lines,
            "non_hyperparam_changed_lines": summary.non_hyperparam_changed_lines,
            "hunks": summary.hunks,
        },
    }
    lock_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def check_lock(lock_path: Path, target_path: Path) -> tuple[bool, str]:
    if not lock_path.exists():
        return False, f"lock file missing: {relpath(lock_path)}"
    payload = json.loads(read_text(lock_path))
    current_sha = sha256_text(read_text(target_path))
    locked_sha = payload.get("target_sha256")
    if current_sha != locked_sha:
        return False, (
            "lock mismatch: current train_gpt.py hash differs from the locked version\n"
            f"locked file: {payload.get('target_path')}\n"
            f"lock file: {relpath(lock_path)}"
        )
    return True, f"lock OK: {relpath(lock_path)} matches current train_gpt.py"


def print_summary(summary: DiffSummary, args: argparse.Namespace) -> int:
    print(f"parent: {relpath(summary.parent_path)}")
    print(f"target: {relpath(summary.target_path)}")
    print(f"hunks: {summary.hunks}")
    print(f"total_changed_lines: {summary.total_changed_lines}")
    print(f"hyperparam_changed_lines: {summary.hyperparam_changed_lines}")
    print(f"hyperparam_envs_changed: {', '.join(summary.hyperparam_envs) if summary.hyperparam_envs else '(none)'}")
    print(f"hyperparam_non_env_lines: {summary.hyperparam_non_env_lines}")
    print(f"non_hyperparam_changed_lines: {summary.non_hyperparam_changed_lines}")

    failures: list[str] = []
    if len(summary.hyperparam_envs) > args.max_env_changes:
        failures.append(
            f"changed {len(summary.hyperparam_envs)} hyperparameter env vars; max allowed is {args.max_env_changes}"
        )
    if summary.hyperparam_non_env_lines > args.max_hparam_misc_lines:
        failures.append(
            "changed non-env lines inside Hyperparameters "
            f"({summary.hyperparam_non_env_lines} > {args.max_hparam_misc_lines})"
        )
    if summary.non_hyperparam_changed_lines > args.max_code_changes:
        failures.append(
            f"changed non-Hyperparameters code lines ({summary.non_hyperparam_changed_lines} > {args.max_code_changes})"
        )
    if summary.total_changed_lines > args.max_total_changed_lines:
        failures.append(
            f"changed too many nonblank lines overall ({summary.total_changed_lines} > {args.max_total_changed_lines})"
        )

    print("\n--- unified diff ---")
    if summary.diff_text:
        print(summary.diff_text, end="" if summary.diff_text.endswith("\n") else "\n")
    else:
        print("(no diff)")

    if failures:
        print("\nFAIL:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPASS: change footprint is within configured surgical limits")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("target", help="leg directory or train_gpt.py to inspect")
    p.add_argument("--parent", help="explicit parent leg directory or parent train_gpt.py")
    p.add_argument("--max-env-changes", type=int, default=1)
    p.add_argument("--max-hparam-misc-lines", type=int, default=0)
    p.add_argument("--max-code-changes", type=int, default=0)
    p.add_argument("--max-total-changed-lines", type=int, default=40)
    p.add_argument("--write-lock", action="store_true", help="write a lock file after analysis")
    p.add_argument("--check-lock", action="store_true", help="verify current file against the saved lock")
    p.add_argument("--lock-file", help="override lock file path")
    return p


def main() -> int:
    args = build_parser().parse_args()
    target_path, parent_path = resolve_target_and_parent(args.target, args.parent)
    lock_path = Path(args.lock_file).resolve() if args.lock_file else default_lock_path(target_path)

    if args.check_lock:
        ok, message = check_lock(lock_path, target_path)
        print(message)
        if not ok:
            return 1

    summary = analyze_diff(parent_path, target_path)
    rc = print_summary(summary, args)

    if rc == 0 and args.write_lock:
        write_lock(lock_path, summary)
        print(f"wrote lock: {relpath(lock_path)}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
