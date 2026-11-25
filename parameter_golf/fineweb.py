"""Path and shard-count rules for the cached FineWeb downloader."""

from __future__ import annotations

from pathlib import Path

DEFAULT_TRAIN_SHARDS = 80


def dataset_dir_for_variant(name: str) -> str:
    if name == "byte260":
        return "fineweb10B_byte260"
    if name.startswith("sp") and name[2:].isdigit():
        return f"fineweb10B_{name}"
    raise ValueError(f"unsupported variant {name!r}; expected byte260 or sp<VOCAB_SIZE>")


def local_path_for_remote(
    relative_path: str,
    *,
    root: Path,
    datasets_dir: Path,
    tokenizers_dir: Path,
    remote_root_prefix: str,
) -> Path:
    remote_path = Path(relative_path)
    if remote_root_prefix and remote_path.parts[:1] == (remote_root_prefix,):
        remote_path = remote_path.relative_to(remote_root_prefix)
    if remote_path.parts[:1] == ("datasets",):
        return datasets_dir.joinpath(*remote_path.parts[1:])
    if remote_path.parts[:1] == ("tokenizers",):
        return tokenizers_dir.joinpath(*remote_path.parts[1:])
    return root / remote_path


def resolve_train_shard_count(
    flag: int | None,
    positional: int | None,
    *,
    default: int = DEFAULT_TRAIN_SHARDS,
) -> int:
    if flag is not None and positional is not None:
        raise ValueError("pass the training shard count with --train-shards or as a positional argument, not both")
    count = default if flag is None and positional is None else flag if flag is not None else positional
    if count is None or count < 0:
        raise ValueError(f"train shard count must be non-negative, got {count}")
    return count


def manifest_shard_counts(dataset_entry: dict) -> tuple[int, int]:
    stats = dataset_entry.get("stats") or {}
    try:
        train_shards = int(stats["files_train"])
        val_shards = int(stats["files_val"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "manifest dataset entry is missing integer stats.files_train and stats.files_val"
        ) from exc
    if train_shards < 0 or val_shards < 0:
        raise ValueError("manifest shard counts must be non-negative")
    return train_shards, val_shards
