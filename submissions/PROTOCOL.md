# Submission Protocol

This is the ONLY process for submitting to openai/parameter-golf.
Every step is required. No improvising. Read this before touching anything.

---

## The One-Line Rule

**Never submit from TEST_LAB. Never push a submission branch to origin.
Never open a PR from origin. Never touch an already-merged PR.**

---

## Prerequisites — Complete BEFORE starting this script

1. Full 8×H100 run complete (seed=444), logs saved
2. Confirmation run complete (seed=300), logs saved
3. Third required run complete (seed=4), logs saved
4. Model beats the current leader BPB on all three seeds (check LEADER.md)
5. `final_model.pt` and `final_model.int6.ptz` saved off pod with unique names
6. All three training logs pulled from pod to this machine

If any of these are missing: STOP. Do not submit a partial.

---

## Step 1 — Build the records folder (on TEST_LAB)

Records path: `records/track_10min_16mb/YYYY-MM-DD_<Name>_8xH100/`

Required files (ALL must exist before Step 2):
```
records/track_10min_16mb/YYYY-MM-DD_<Name>_8xH100/
  submission.json        ← see template: submissions/templates/submission_neural.json
  train_gpt.py           ← the EXACT file that ran (vault copy for neural)
  train_seed444.log      ← full log from seed=444 run
  train_seed300.log      ← full log from seed=300 run
  train_seed4.log        ← full log from seed=4 run
  README.md              ← results table + reproduce instructions
```

Optional (add if you have them):
```
  gate_seed444.log       ← 1-GPU gate log
```

Neural: copy train_gpt.py from vault/, not from neural/<leg>/
Crawler: copy train_gpt.py from crawler/<champion_leg>/

### submission.json — required fields

```json
{
  "author": "Frosty40",
  "github_id": "newjordan",
  "name": "<short PR name>",
  "blurb": "<one sentence describing the architecture change>",
  "date": "<YYYY-MM-DDT00:00:00Z>",
  "seed_444": {
    "val_bpb": <round to 4 decimal>,
    "val_bpb_exact": <full precision from log>,
    "steps": <int>,
    "train_time_s": 600,
    "bytes_total": <int from log>
  },
  "seed_300": {
    "val_bpb": <round to 4 decimal>,
    "val_bpb_exact": <full precision from log>,
    "steps": <int>,
    "bytes_total": <int from log>,
    "train_time_s": 600
  },
  "seed_4": {
    "val_bpb": <round to 4 decimal>,
    "val_bpb_exact": <full precision from log>,
    "steps": <int>,
    "bytes_total": <int from log>,
    "train_time_s": 600
  },
  "val_bpb": <mean of all seeds, 4 decimal>,
  "bytes_total": <MAX bytes_total across seeds (444/300/4)>,
  "bytes_code": <len(train_gpt.py.encode('utf-8')) — check log output>,
  "hardware": "8xH100 SXM"
}
```

Validation checklist for submission.json:
- [ ] `bytes_total` (max across seeds) is ≤ 16,000,000 bytes (16MB hard cap)
- [ ] `bytes_code` matches the `Code size:` line in your training log
- [ ] `val_bpb_exact` matches the `final_sliding_window_exact val_bpb=` line in your log
- [ ] `date` is the date of the run, not today

---

## Step 2 — Run the validation script

```bash
bash submissions/validate.sh records/track_10min_16mb/YYYY-MM-DD_<Name>_8xH100/
```

This checks all four required files exist, validates submission.json fields,
and confirms bytes_total is legal. Fix any errors before continuing.

---

## Step 3 — Create a clean submission branch (private, never origin)

```bash
# From TEST_LAB after validate passes
git fetch upstream
git checkout -b submission/<name> upstream/main
git checkout TEST_LAB -- records/track_10min_16mb/YYYY-MM-DD_<Name>_8xH100/
# Example: git checkout -b submission/rascal-iii upstream/main
```

The branch name should be short and match the PR name. Use kebab-case.
The submission branch must be based on `upstream/main` so the PR diff stays clean.

Commit ONLY the records folder. Nothing else:
```bash
git add records/track_10min_16mb/YYYY-MM-DD_<Name>_8xH100/
git commit -m "Add <Name> submission — <BPB> BPB, <size>MB"
```

---

## Step 4 — Push to fork1 (NOT origin)

```bash
git push fork1 submission/<name>
```

`fork1` = https://github.com/newjordan/parameter-golf-1 (the public competition fork)
`origin` = https://github.com/newjordan/parameter-golf (our private lab — NEVER gets submission branches)

Verify it's on fork1: https://github.com/newjordan/parameter-golf-1/branches

---

## Step 5 — Open the PR

```bash
gh pr create \
  --repo openai/parameter-golf \
  --head "newjordan:submission/<name>" \
  --title "<Name> — <BPB> val_bpb (seed 444)" \
  --body "$(cat submissions/templates/pr_body_template.md)"
```

Edit the body template BEFORE running this. Replace all `<placeholders>`.
Include seed 4 in the PR Results table (seeds 444, 300, and 4 are required).

PR title format: `<ModelName> — <exact_bpb> val_bpb (seed 444)`
Example: `Rascal III — 1.10812345 val_bpb (seed 444)`

---

## Step 6 — After the PR is open

1. Copy the PR URL and save it in the relevant RESULTS.md
2. Update LEADER.md (neural or crawler) with the new score
3. Switch back to TEST_LAB: `git checkout TEST_LAB`
4. **Never touch the submission branch again.** If the PR needs a fix, ask first.

---

## What kills PRs (learned the hard way)

| Mistake | Cost |
|---------|------|
| Missing submission.json | PR closed, leaderboard position lost (PR #674) |
| Missing training logs | PR closed |
| Wrong train_gpt.py (wrong file, wrong size) | Invalid submission, score rejected |
| bytes_total > 16MB | Disqualified |
| Submitting with TEST_LAB ancestry | Huge noisy PR diff, low reviewer trust |
| Touching a merged PR | Reopens old issues, breaks submission record |
| COPRIME_MAX_LOADED_SHARDS != 1 | Wrong training trajectory, worse BPB |

---

## Quick Reference

| Repo | Remote | Purpose |
|------|--------|---------|
| newjordan/parameter-golf | `origin` | Daily lab work, TEST_LAB branch |
| newjordan/parameter-golf-1 | `fork1` | Submission branches ONLY |
| openai/parameter-golf | `upstream` | Competition target — PRs go here |

Branch flow: `TEST_LAB (prep + validate)` → `submission/<name>` from `upstream/main` → push `fork1` → PR to `upstream/main`
