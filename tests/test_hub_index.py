import json
from pathlib import Path

from parameter_golf import ARTIFACT_CAP_BYTES
from parameter_golf.hub_index import build_index, render_index

ROOT = Path(__file__).resolve().parents[1]


def _write_submission(folder: Path, name: str, val_bpb: float, bytes_total: int, author: str = "tester") -> None:
    folder.mkdir(parents=True)
    payload = {
        "name": name,
        "author": author,
        "github_id": author,
        "date": "2026-03-01",
        "val_bpb": val_bpb,
        "bytes_total": bytes_total,
    }
    (folder / "submission.json").write_text(json.dumps(payload), encoding="utf-8")


def test_index_ranks_submissions_and_drops_secrets(tmp_path):
    records = tmp_path / "records" / "track_10min_16mb"
    _write_submission(records / "2026-03-01_baseline", "Naive Baseline", 1.5, 15_000_000, "baseline")
    (records / "2026-03-01_baseline" / "submission.json").write_text(
        json.dumps(
            {
                "name": "Naive Baseline",
                "author": "baseline",
                "github_id": "baseline",
                "date": "2026-03-01",
                "val_bpb": 1.5,
                "bytes_total": 15_000_000,
            }
        ),
        encoding="utf-8",
    )
    # The folder name has to contain NaiveBaseline for the delta baseline.
    baseline_dir = records / "2026-03-01_NaiveBaseline"
    baseline_dir.mkdir()
    (baseline_dir / "submission.json").write_text(
        json.dumps(
            {
                "name": "Naive Baseline",
                "author": "baseline",
                "date": "2026-03-01",
                "val_bpb": 1.5,
                "bytes_total": 15_000_000,
            }
        ),
        encoding="utf-8",
    )
    better = records / "2026-03-02_better"
    _write_submission(better, "Better", 1.2, 14_000_000, "ada")
    over = records / "2026-03-02_over"
    _write_submission(over, "Over Cap", 1.1, ARTIFACT_CAP_BYTES + 1, "ada")
    secret = records / "2026-03-02_secret"
    secret.mkdir()
    (secret / "ssh_info.env").write_text("INSTANCE_API_KEY=abcdef123456\n", encoding="utf-8")
    (secret / "train.log").write_text(
        "step:1 val_bpb:1.25\ninstance_api_key=abcdef1234567890 token stays out\n",
        encoding="utf-8",
    )

    payload = build_index(tmp_path)
    blob = render_index(payload)
    assert "abcdef1234567890" not in blob
    assert "/home/" not in blob
    assert "ssh://" not in blob
    assert all(not record["path"].startswith("/") for record in payload["records"])
    assert all(record["category"] != "env" for record in payload["records"])
    assert not any(record["rel_path"].endswith(".env") for record in payload["records"])

    leaders = payload["independent_rankings"]["best_bpb"]
    assert [row["value"] for row in leaders] == [1.1, 1.2, 1.5, 1.5]
    assert payload["records"]
    over_row = next(record for record in payload["records"] if record["run_tag"] == "Over Cap")
    assert over_row["status"] == "warn"
    assert any("cap" in note for note in over_row["notes"])
    log_row = next(record for record in payload["records"] if record["rel_path"].endswith("train.log"))
    assert log_row["metrics"]["val_bpb"] == 1.25
    assert "abcdef1234567890" not in log_row["snippet"]
    improved = [item for item in payload["ablations"] if item["candidate_label"] == "Better"]
    assert improved and improved[0]["delta"] < 0
    assert payload["generated_at"] == "2026-03-02T00:00:00Z"


def test_committed_index_matches_records_and_has_no_private_lab_dump():
    committed_path = ROOT / "hub_index.json"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    built = json.loads(render_index(build_index(ROOT)))
    assert built == committed
    blob = committed_path.read_text(encoding="utf-8")
    assert "instance_api_key" not in blob
    assert "/home/" not in blob
    assert "ssh://" not in blob
    assert committed["source_roots"] == ["records"]
    assert committed["artifact_cap_bytes"] == ARTIFACT_CAP_BYTES
    assert committed["counts"]["submissions"] == 23
    assert committed["counts"]["total_records"] == 120
    assert committed["counts"]["by_status"]["ok"] == 23
    assert "warn" not in committed["counts"]["by_status"]
    mlp = next(
        record
        for record in committed["records"]
        if record["rel_path"].endswith("2026-03-19_MLP3x_QAT_Int6_SlidingWindow/submission.json")
    )
    assert mlp["status"] == "ok"
    assert mlp["run_tag"] == "11L_MLP3x_WD04_Int6QAT_Zstd22_SlidingWindow"
    assert mlp["metrics"]["model_size_bytes"] == 15_427_455
    assert committed["independent_rankings"]["best_bpb"][0]["value"] == 1.1194
    assert committed["independent_rankings"]["best_base_model"]
    assert committed["independent_rankings"]["best_base_model"][0]["value"] <= committed["independent_rankings"]["best_base_model"][-1]["value"]
