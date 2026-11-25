import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    re.compile(r"instance_api_key", re.IGNORECASE),
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"ssh://"),
)


def test_published_records_and_hub_have_no_private_lab_secrets():
    targets = [ROOT / "hub_index.json", ROOT / "README.md", ROOT / "index.html", ROOT / "app.js"]
    targets.extend(path for path in (ROOT / "records").rglob("*") if path.is_file())
    targets.extend(path for path in (ROOT / "data").rglob("*") if path.is_file())
    offenders = []
    for path in targets:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pt", ".bin"}:
            continue
        text = path.read_bytes()[:2_000_000].decode("utf-8", errors="ignore")
        for pattern in FORBIDDEN:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(ROOT)} matched {pattern.pattern}")
    assert offenders == []
