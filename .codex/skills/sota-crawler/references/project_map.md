# SOTA Crawler Project Map

## Scope

This workspace is the crawler-side lab for the Parameter Golf challenge. The active optimization target is low `int6_sw_bpb` with artifact size safely under the 16MB cap.

## Canonical files

- `LEADER.md`: current crawler champion, baseline leg, promotion threshold.
- `PIPELINE.md`: ranked hypothesis queue and closed/dead ideas.
- `SCIENCE.md`: experiment history, measured deltas, and interpretation.
- `CLAUDE.md`: repo-local operating protocol.
- `LAB_PROTOCOL.md`: lab discipline and cost rules.

## Directory intent

- `legs/`: active experiment legs. Each leg should contain `hypothesis.md`, `train_gpt.py`, `gate.sh`, `ablation.md`, and `RESULTS.md`.
- `records/`: frozen competition artifacts and accepted-style packaging examples.
- `submissions/`: validation and PR packaging workflow.
- `scripts/`: scaffolding, pod bootstrap, environment setup, and runner helpers.
- `experiments/`: active scratch work outside the main crawler leg line.
- `junkyard/`: legacy or exploratory history; reference only unless the user explicitly revives it.
- `data/`: symlinked dataset area; do not modify.

## Current crawler baseline

From `LEADER.md` and `CLAUDE.md`:

- Leader leg: `legs/2026-03-29_BW5/`
- Primary metric: `1.18672385` BPB on seed `444`
- Mean over two seeds: `1.18715`
- Baseline invariants: `COMPILE_FULLGRAPH=1`, `CRAWLER_MLP_CHOKE_DIM=0`, `CRAWLER_LOOP_ROPE_SCALES=9,1,1`, `SKIP_GPTQ=1`

Always re-check `LEADER.md` before using these values; the skill should track the file, not memory.

## Standard experiment workflow

1. Orient on `LEADER.md`, `PIPELINE.md`, `SCIENCE.md`.
2. Confirm one test variable and an explicit parent leg.
3. Scaffold with `bash scripts/new_leg.sh <name>` when starting a fresh leg.
4. Run the 1-GPU gate first.
5. Record the gate in `ablation.md`.
6. Only then consider the 8xH100 600s run on seed `444`.
7. If it beats the leader and size is legal, confirm on seed `300`.
8. Record the verdict in `RESULTS.md` and update leader-facing docs only after confirmation.

Before step 4, run:

```bash
python3 scripts/leg_diff_guard.py legs/<leg_name>
```

Optional lock flow for extra-sensitive work:

```bash
python3 scripts/leg_diff_guard.py legs/<leg_name> --write-lock
python3 scripts/leg_diff_guard.py legs/<leg_name> --check-lock
```

The guard infers the parent from the leg metadata, counts changed Hyperparameters env vars, counts non-Hyperparameters code edits, prints a unified diff, and fails if the edit footprint exceeds the configured surgical limits.

## Evaluation heuristics

- Small proxy wins are suspect until they survive the gate.
- Step time regressions matter because the wallclock budget is fixed.
- A win that breaks the size cap is not a promotion.
- Stacked positive micro-signals can still regress in a full run; treat composition skeptically.
- Closed or dead ideas in `PIPELINE.md` and `SCIENCE.md` should not be retried unless the user explicitly reopens them with a new mechanism.

## Submission workflow

Submission work is separate from experiment work.

- Validate from `submissions/validate.sh` before any branch or PR work.
- Use `submissions/templates/submission_crawler.json` for metadata shape.
- Keep submission branches separate from day-to-day lab branches.
- Do not trust stale narrative notes when a warning banner says they are invalidated; prefer current scripts, templates, and repository state.
