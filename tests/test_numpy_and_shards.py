import numpy as np
import pytest

from parameter_golf.numpy_compat import cast_array
from parameter_golf.shards import read_shard, write_shard


def test_numpy_major_is_at_least_2():
    assert int(np.__version__.split(".", maxsplit=1)[0]) >= 2


def test_numpy2_rejects_array_copy_false_when_a_cast_is_required():
    values = np.array([1, 2], dtype=np.uint16)
    with pytest.raises(ValueError, match="Unable to avoid copy"):
        np.array(values, dtype=np.int32, copy=False)
    copied = values.astype(np.int32, copy=False)
    assert copied.dtype == np.dtype(np.int32)
    assert copied is not values


def test_cast_array_copies_only_when_the_dtype_changes():
    values = np.array([4, 5, 6], dtype=np.uint16)
    same = cast_array(values, np.uint16)
    assert same.dtype == np.dtype("<u2")
    assert same.tolist() == [4, 5, 6]
    widened = cast_array(values, np.int32)
    assert widened.dtype == np.dtype(np.int32)
    assert widened.tolist() == [4, 5, 6]


def test_shard_roundtrip_and_size_mismatch(tmp_path):
    path = tmp_path / "fineweb_train_000000.bin"
    tokens = np.array([0, 4, 259, 65535], dtype=np.int32)
    write_shard(path, tokens)
    loaded = read_shard(path)
    assert loaded.dtype == np.dtype("<u2")
    assert loaded.tolist() == [0, 4, 259, 65535]

    blob = path.read_bytes()
    path.write_bytes(blob[:-2])
    with pytest.raises(ValueError, match="Shard size mismatch"):
        read_shard(path)


def test_shard_rejects_token_ids_outside_uint16(tmp_path):
    with pytest.raises(ValueError, match="uint16"):
        write_shard(tmp_path / "bad.bin", np.array([70000], dtype=np.int32))
