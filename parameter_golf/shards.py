"""Read and write the challenge ``fineweb_*.bin`` shard format.

The on-disk layout is a 256-int32 little-endian header followed by uint16
token ids. Header fields used by the starter scripts are:

- ``[0]`` magic ``20240520``
- ``[1]`` version ``1``
- ``[2]`` token count
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from parameter_golf.numpy_compat import cast_array

DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
HEADER_INTS = 256
HEADER_DTYPE = np.dtype("<i4")
TOKEN_DTYPE = np.dtype("<u2")


def header_nbytes() -> int:
    return HEADER_INTS * HEADER_DTYPE.itemsize


def as_uint16_tokens(tokens) -> np.ndarray:
    array = np.asarray(tokens)
    if array.size and (np.any(array < 0) or np.any(array >= 2**16)):
        raise ValueError("token id does not fit in uint16")
    return cast_array(array, TOKEN_DTYPE)


def write_shard(path: Path, tokens) -> None:
    toks = as_uint16_tokens(tokens)
    if toks.size >= 2**31:
        raise ValueError("token count too large")
    header = np.zeros(HEADER_INTS, dtype=HEADER_DTYPE)
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = int(toks.size)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(toks.tobytes())


def read_shard(path: Path) -> np.ndarray:
    path = Path(path)
    header_bytes = header_nbytes()
    token_bytes = TOKEN_DTYPE.itemsize
    header = np.fromfile(path, dtype=HEADER_DTYPE, count=HEADER_INTS)
    if header.size != HEADER_INTS or int(header[0]) != DATAFILE_MAGIC or int(header[1]) != DATAFILE_VERSION:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if num_tokens < 0:
        raise ValueError(f"Negative token count in {path}")
    expected_size = header_bytes + num_tokens * token_bytes
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"Shard size mismatch for {path}: expected {expected_size} bytes, found {actual_size}")
    tokens = np.fromfile(path, dtype=TOKEN_DTYPE, count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens
