"""Write or check ``hub_index.json``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from parameter_golf.hub_index import build_index, render_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the research hub index from records/")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None, help="Output path. Defaults to <root>/hub_index.json.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 when the output file does not match a fresh build.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = (args.out or (root / "hub_index.json")).resolve()
    payload = build_index(root)
    text = render_index(payload)
    counts = payload["counts"]
    if args.check:
        if not out.is_file():
            print(f"missing index: {out}", file=sys.stderr)
            return 1
        existing = out.read_text(encoding="utf-8")
        if existing != text:
            print(f"{out} does not match the builder output. Run python -m parameter_golf.build_index", file=sys.stderr)
            return 1
        print(
            f"{out} matches the builder "
            f"(records={counts['total_records']} submissions={counts['submissions']})"
        )
        return 0
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} records={counts['total_records']} submissions={counts['submissions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
