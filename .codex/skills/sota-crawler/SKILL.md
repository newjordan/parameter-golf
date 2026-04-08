---
name: sota-crawler
description: Use when working in /home/frosty40/sota_crawler on Bandit_Wagon crawler research, experiment legs, gates, full runs, ablations, results, or submission prep. Enforces the lab's one-variable-per-test discipline, gate-before-8x workflow, seed protocol, and project file map.
---

# SOTA Crawler

This skill is for the local `sota_crawler` workspace. Use it before proposing or editing crawler experiments, interpreting ablations, or preparing crawler submission artifacts.

## Start Here

Read these files first unless the task is narrowly scoped to one leg:

```bash
sed -n '1,120p' LEADER.md
sed -n '1,220p' PIPELINE.md
sed -n '1,240p' SCIENCE.md
sed -n '1,220p' CLAUDE.md
```

Choose extra context by task:

- New leg or hypothesis: read [references/project_map.md](references/project_map.md) and `scripts/new_leg.sh`.
- Existing leg update: read that leg's `hypothesis.md`, `ablation.md`, `RESULTS.md`, and parent `train_gpt.py`.
- Submission prep: read `submissions/CLAUDE.md`, `submissions/validate.sh`, and templates in `submissions/templates/`.

## Non-Negotiables

- Change one variable per test. If more than one behaviorally relevant variable moved, stop and split the experiment.
- Gate before any expensive run. Minimum trusted signal is the 1-GPU 2000-step gate.
- Primary seed is `444`. Confirmation seed is `300`. Do not use `1337` for new work.
- Base crawler leader is `legs/2026-03-29_BW5/` unless `LEADER.md` says otherwise.
- `COMPILE_FULLGRAPH=1` is baseline for BW5+ crawler legs.
- Keep artifact size under `16,000,000` bytes before promoting or packaging a submission.
- Never overwrite prior test artifacts. New leg, new files, clear names.
- Treat `records/` as frozen references unless the task is explicit submission packaging.

## Working Rules

### When creating a new leg

1. Confirm the intended leg name with the user when naming matters.
2. Scaffold from the leader with `bash scripts/new_leg.sh <name>`.
3. Write `hypothesis.md` first.
4. Make the single change in the copied `train_gpt.py`.
5. Keep `gate.sh`, `ablation.md`, and `RESULTS.md` aligned with the same parent and hypothesis.
6. Run the diff guard before any gate or pod run:

```bash
python3 scripts/leg_diff_guard.py legs/<leg_name>
python3 scripts/leg_diff_guard.py legs/<leg_name> --write-lock
python3 scripts/leg_diff_guard.py legs/<leg_name> --check-lock
```

Use the default thresholds for surgical one-variable work. Only relax them intentionally.

### When editing an existing leg

1. Verify the parent leg and the single variable under test.
2. Diff against the parent before claiming the leg is valid.
3. Preserve logs, result tables, and recorded deltas unless the user explicitly asks to revise history.

### When reviewing a result

Check all four dimensions together:

- `int6_sw_bpb`
- artifact bytes
- step time / throughput
- seed behavior (`444` vs `300`)

Do not promote from proxy enthusiasm alone. The project docs explicitly note that short proxy deltas can inflate far beyond full-run reality.

## Submission Caution

The `submissions/` area is high-stakes. Prefer live validation over stale prose.

Read in this order:

1. `submissions/CLAUDE.md`
2. `submissions/validate.sh`
3. `submissions/templates/submission_crawler.json`
4. `submissions/templates/pr_body_template.md`

Treat warning-bannered notes in `submissions/*.md` as advisory unless confirmed by current scripts and repo state.

## Repo Map

Use [references/project_map.md](references/project_map.md) for the stable map, workflow, and known crawler-specific invariants distilled from project docs.
