"""Build the research-hub index from checked-in ``records/`` files.

The index stores repo-relative paths only. Environment files and credential
assignments are dropped or redacted so a regenerated index cannot republish
the private lab dump that used to live in ``hub_index.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from parameter_golf import ARTIFACT_CAP_BYTES
from parameter_golf.redact import is_sensitive_path, redact_text

_DATE_IN_PATH = re.compile(r"(\d{4}-\d{2}-\d{2})")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z)?")
_VAL_BPB = re.compile(r"\bval_bpb\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")
_ERROR_LINE = re.compile(
    r"(?im)^(?:\[[^\]]+\]:\s*)?(?:Traceback \(most recent call last\):|torch\.OutOfMemoryError\b|"
    r"CUDA out of memory|ChildFailedError:|Shard size mismatch\b)"
)
_KEYWORD_GROUPS = (
    ("traceback", "traceback"),
    ("oom", "oom"),
    ("shard size mismatch", "shard size mismatch"),
    ("cuda error", "cuda error"),
    ("valueerror", "valueerror"),
    ("runtimeerror", "runtimeerror"),
    ("promote", "promote"),
    ("proxy", "proxy"),
    ("regression", "regression"),
)
_BPB_KEYS = ("val_bpb", "mean_val_bpb")
_BYTE_KEYS = ("bytes_total", "artifact_bytes", "artifact_size_bytes", "artifact_bytes_mean")
_TEXT_LIMIT = 2_000_000
_SNIPPET_LIMIT = 420


def build_index(root: Path) -> dict:
    root = Path(root)
    records_root = root / "records"
    records: list[dict] = []
    if records_root.is_dir():
        for path in sorted(p for p in records_root.rglob("*") if p.is_file()):
            record = _record_from_path(root, path)
            if record is not None:
                records.append(record)

    submissions = [record for record in records if record["category"] == "submission"]
    rankings = {
        "best_bpb": _rank_bpb(submissions, track=None),
        "best_base_model": _rank_bpb(submissions, track="track_10min_16mb"),
        "lowest_file_size": _rank_bytes(submissions),
    }
    ablations = _ablations(submissions)
    dates = [record["timestamp_hint"] for record in records if _ISO_DATE.fullmatch(record.get("timestamp_hint") or "")]
    generated_at = max((_normalize_date(value) for value in dates), default="")
    payload = {
        "generated_at": generated_at,
        "generated_at_source": "newest submission or record date in records/",
        "source_roots": ["records"],
        "artifact_cap_bytes": ARTIFACT_CAP_BYTES,
        "counts": _counts(records, ablations),
        "hypothesis": _hypothesis(rankings, ablations),
        "favorite_considerations": _favorites(rankings),
        "independent_rankings": rankings,
        "ablations": ablations,
        "charts": {
            "status_distribution": _status_distribution(records),
            "top_ablation_deltas": _chart_deltas(ablations),
            "timeline": _timeline(records),
        },
        "records": records,
    }
    return payload


def render_index(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _record_from_path(root: Path, path: Path) -> dict | None:
    if is_sensitive_path(path):
        return None
    category = _category(path)
    if category is None:
        return None
    rel_path = path.relative_to(root).as_posix()
    text = _read_text(path)
    redacted = redact_text(text)
    submission = _load_submission(path, text) if category == "submission" else None
    metrics: dict[str, float] = {}
    notes: list[str] = []
    bpb_key, bpb = (None, None)
    byte_count = None
    author = ""
    run_tag = path.stem
    if submission is not None:
        bpb_key, bpb = _submission_bpb(submission)
        byte_count = _submission_bytes(submission)
        author = str(submission.get("github_id") or submission.get("author") or "").strip()
        run_tag = str(submission.get("name") or submission.get("run_name") or path.parent.name)
        if bpb is not None:
            metrics["val_bpb"] = bpb
        loss = submission.get("val_loss")
        if isinstance(loss, (int, float)) and not isinstance(loss, bool):
            if bpb is None or abs(float(loss) - bpb) > 1e-6:
                metrics["val_loss"] = float(loss)
        if byte_count is not None:
            metrics["model_size_bytes"] = float(byte_count)
    elif category in {"run_log", "tsv_metric"}:
        found = _VAL_BPB.findall(redacted)
        if found:
            metrics["val_bpb"] = float(found[-1])

    status = _status(category, redacted, bpb, byte_count, notes)
    timestamp = _timestamp(rel_path, submission)
    record = {
        "id": f"{category}:{hashlib.sha256(rel_path.encode('utf-8')).hexdigest()[:16]}",
        "path": rel_path,
        "rel_path": rel_path,
        "category": category,
        "track": _track(rel_path),
        "experiment_group": path.parent.name,
        "run_tag": run_tag,
        "author": author,
        "timestamp_hint": timestamp,
        "metrics": metrics,
        "status": status,
        "notes": notes,
        "snippet": _snippet(redacted, category),
        "keywords": _keywords(redacted),
    }
    if bpb_key:
        record["bpb_source"] = bpb_key
    return record


def _category(path: Path) -> str | None:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pt", ".bin", ".ptz", ".safetensors"}:
        return None
    if name == "submission.json":
        return "submission"
    if name == "readme.md" or suffix == ".md":
        return "report"
    if suffix == ".py":
        return "script"
    if suffix in {".log", ".txt"}:
        return "run_log"
    if suffix == ".tsv":
        return "tsv_metric"
    if suffix == ".json":
        return "summary"
    return None


def _read_text(path: Path) -> str:
    data = path.read_bytes()[:_TEXT_LIMIT]
    return data.decode("utf-8", errors="replace")


def _load_submission(path: Path, text: str) -> dict | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _submission_bpb(payload: dict) -> tuple[str | None, float | None]:
    for key in _BPB_KEYS:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return key, float(value)
    seeds = payload.get("seed_results")
    if isinstance(seeds, dict):
        values = []
        for item in seeds.values():
            if isinstance(item, dict):
                seed_bpb = item.get("val_bpb")
                if isinstance(seed_bpb, (int, float)) and not isinstance(seed_bpb, bool):
                    values.append(float(seed_bpb))
        if values:
            return "seed_results.val_bpb", sum(values) / len(values)
    return None, None


def _submission_bytes(payload: dict) -> int | None:
    for key in _BYTE_KEYS:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def _status(category: str, text: str, bpb: float | None, byte_count: int | None, notes: list[str]) -> str:
    if category == "submission":
        if bpb is None:
            notes.append("submission.json has no val_bpb or mean_val_bpb")
            return "error"
        if byte_count is None:
            notes.append("artifact byte count is missing")
            return "warn"
        if byte_count > ARTIFACT_CAP_BYTES:
            notes.append(f"artifact is over the {ARTIFACT_CAP_BYTES} byte cap")
            return "warn"
        return "ok"
    if category in {"run_log", "tsv_metric"} and _ERROR_LINE.search(text):
        notes.append("log contains a traceback, OOM, or shard-size failure")
        return "warn" if "val_bpb" in text else "error"
    return "unknown"


def _timestamp(rel_path: str, submission: dict | None) -> str:
    if submission is not None:
        raw = submission.get("date")
        if isinstance(raw, str) and _normalize_date(raw):
            return _normalize_date(raw)
    match = _DATE_IN_PATH.search(rel_path)
    if match:
        return match.group(1)
    return ""


def _normalize_date(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return f"{value}T00:00:00Z"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        return value
    return ""


def _track(rel_path: str) -> str:
    parts = rel_path.split("/")
    if len(parts) >= 2 and parts[0] == "records":
        return parts[1]
    return "records"


def _snippet(text: str, category: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if category in {"run_log", "tsv_metric"}:
        interesting = [
            line
            for line in lines
            if "val_bpb" in line.lower() or "traceback" in line.lower() or "error" in line.lower()
        ]
        chosen = interesting[-3:] if interesting else lines[:3]
    else:
        chosen = lines[:4]
    snippet = " | ".join(chosen)
    if len(snippet) > _SNIPPET_LIMIT:
        return snippet[: _SNIPPET_LIMIT - 3] + "..."
    return snippet


def _keywords(text: str) -> list[str]:
    lowered = text.lower()
    return [label for needle, label in _KEYWORD_GROUPS if needle in lowered]


def _rank_bpb(submissions: list[dict], track: str | None) -> list[dict]:
    rows = []
    for record in submissions:
        if track is not None and record["track"] != track:
            continue
        value = record["metrics"].get("val_bpb")
        if value is None:
            continue
        rows.append(_ranking_row(record, "best_bpb" if track is None else "best_base_model", "Best BPB" if track is None else "Best Base Model", "val_bpb", value))
    rows.sort(key=lambda row: (row["value"], row["rel_path"]))
    return rows


def _rank_bytes(submissions: list[dict]) -> list[dict]:
    rows = []
    for record in submissions:
        value = record["metrics"].get("model_size_bytes")
        if value is None:
            continue
        rows.append(_ranking_row(record, "lowest_file_size", "Lowest File Size", "model_size_bytes", value))
    rows.sort(key=lambda row: (row["value"], row["rel_path"]))
    return rows


def _ranking_row(record: dict, category: str, label: str, metric_used: str, value: float) -> dict:
    return {
        "category": category,
        "label": label,
        "metric_key": category,
        "metric_used": metric_used,
        "value": value,
        "run_tag": record["run_tag"],
        "experiment_group": record["experiment_group"],
        "status": record["status"],
        "rel_path": record["rel_path"],
        "id": record["id"],
        "track": record["track"],
        "author": record["author"],
    }


def _favorites(rankings: dict) -> list[dict]:
    chosen = []
    for key, label in (
        ("best_bpb", "Best BPB"),
        ("best_base_model", "Best Base Model"),
        ("lowest_file_size", "Lowest File Size"),
    ):
        rows = rankings.get(key) or []
        if not rows:
            continue
        row = dict(rows[0])
        row["label"] = label
        chosen.append(row)
    return chosen


def _ablations(submissions: list[dict]) -> list[dict]:
    baseline = next((record for record in submissions if "NaiveBaseline" in record["rel_path"]), None)
    if baseline is None or "val_bpb" not in baseline["metrics"]:
        return []
    baseline_value = baseline["metrics"]["val_bpb"]
    rows = []
    for record in submissions:
        if record["id"] == baseline["id"] or "val_bpb" not in record["metrics"]:
            continue
        candidate = record["metrics"]["val_bpb"]
        delta = candidate - baseline_value
        if delta < -1e-6:
            verdict = "improved"
        elif delta > 1e-6:
            verdict = "regressed"
        else:
            verdict = "flat"
        rows.append(
            {
                "title": record["run_tag"],
                "group": record["experiment_group"],
                "summary": (
                    f"{record['run_tag']} val_bpb { _format_metric(candidate) } versus Naive Baseline "
                    f"{_format_metric(baseline_value)} (delta {_format_metric(delta)})."
                ),
                "primary_metric": "val_bpb",
                "baseline_label": baseline["run_tag"],
                "baseline_value": baseline_value,
                "candidate_label": record["run_tag"],
                "candidate_value": candidate,
                "delta": delta,
                "verdict": verdict,
                "rel_path": record["rel_path"],
            }
        )
    rows.sort(key=lambda row: (row["delta"], row["rel_path"]))
    return rows


def _chart_deltas(ablations: list[dict]) -> list[dict]:
    if len(ablations) <= 8:
        chosen = ablations
    else:
        chosen = ablations[:4] + ablations[-4:]
    return [
        {"label": item["title"], "delta": item["delta"], "primary_metric": item["primary_metric"]}
        for item in chosen
    ]


def _status_distribution(records: list[dict]) -> list[dict]:
    counts = Counter(record["status"] for record in records)
    return [{"name": name, "value": counts[name]} for name in ("ok", "warn", "error", "unknown") if counts[name]]


def _timeline(records: list[dict]) -> list[dict]:
    buckets: dict[str, Counter] = {}
    for record in records:
        hint = record.get("timestamp_hint") or ""
        match = _DATE_IN_PATH.search(hint)
        day = match.group(1) if match else "unknown"
        buckets.setdefault(day, Counter())[record["status"]] += 1
    rows = []
    for day in sorted(buckets):
        counts = buckets[day]
        rows.append(
            {
                "day": day,
                "ok": counts["ok"],
                "warn": counts["warn"],
                "error": counts["error"],
                "unknown": counts["unknown"],
            }
        )
    return rows


def _counts(records: list[dict], ablations: list[dict]) -> dict:
    by_category = Counter(record["category"] for record in records)
    by_status = Counter(record["status"] for record in records)
    return {
        "total_records": len(records),
        "by_category": dict(sorted(by_category.items())),
        "by_status": dict(sorted(by_status.items())),
        "with_metrics": sum(1 for record in records if record["metrics"]),
        "with_errors": by_status["error"],
        "with_promote_notes": sum(1 for record in records if "promote" in record["keywords"]),
        "ablations": len(ablations),
        "submissions": by_category["submission"],
    }


def _hypothesis(rankings: dict, ablations: list[dict]) -> dict:
    if rankings.get("best_base_model"):
        leader = rankings["best_base_model"][0]
        scope = "10-minute"
    elif rankings.get("best_bpb"):
        leader = rankings["best_bpb"][0]
        scope = "indexed"
    else:
        return {
            "current_hypothesis": "No submission.json with val_bpb is indexed yet.",
            "supporting_signal": "Add a record under records/ and rebuild the hub index.",
            "contradictory_signal": "No contradictory submission is indexed.",
            "next_test": "Index the checked-in records before changing the starter script.",
        }
    improved = [item for item in ablations if item["verdict"] == "improved"]
    regressed = [item for item in ablations if item["verdict"] == "regressed"]
    support = improved[0]["summary"] if improved else "No baseline delta is available."
    contradiction = regressed[-1]["summary"] if regressed else "No checked-in submission is worse than the Naive Baseline val_bpb."
    return {
        "current_hypothesis": (
            f"{leader['run_tag']} leads the {scope} submissions at val_bpb {_format_metric(leader['value'])}."
        ),
        "supporting_signal": support,
        "contradictory_signal": contradiction,
        "next_test": (
            "Reproduce the leading record from its records folder on the official 8xH100 budget "
            "before editing the starter script."
        ),
    }


def _format_metric(value: float) -> str:
    return format(value, ".10g")
