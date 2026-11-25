"""Redact credential-shaped text before it is written into the research hub."""

from __future__ import annotations

import re
from pathlib import Path

_ASSIGNMENT = re.compile(
    r"(?i)\b(?:instance_api_key|api[_-]?key|secret|password|access[_-]?token|private[_-]?key|authorization)"
    r"\s*['\"]?\s*[:=]\s*['\"]?[^\s|'\"\\]{6,}"
)
_SSH_URL = re.compile(r"ssh://[^\s|]+")
_PEM = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
)

_SKIP_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
}
_SKIP_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".p8"}


def is_sensitive_path(path: Path) -> bool:
    name = path.name.lower()
    if name in _SKIP_NAMES or name.startswith(".env") or name.endswith(".env"):
        return True
    return path.suffix.lower() in _SKIP_SUFFIXES


def redact_text(text: str) -> str:
    redacted = _PEM.sub("[redacted]", text)
    redacted = _SSH_URL.sub("[redacted]", redacted)
    redacted = _ASSIGNMENT.sub("[redacted]", redacted)
    return redacted
