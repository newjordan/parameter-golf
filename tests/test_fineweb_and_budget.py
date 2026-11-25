import pytest

from parameter_golf.budget import cuda_grad_accum_steps, require_divisible_token_budget
from parameter_golf.fineweb import (
    dataset_dir_for_variant,
    local_path_for_remote,
    manifest_shard_counts,
    resolve_train_shard_count,
)


def test_variant_directories():
    assert dataset_dir_for_variant("sp1024") == "fineweb10B_sp1024"
    assert dataset_dir_for_variant("byte260") == "fineweb10B_byte260"
    with pytest.raises(ValueError):
        dataset_dir_for_variant("gpt2")


def test_remote_paths_strip_the_datasets_prefix(tmp_path):
    root = tmp_path / "data"
    datasets = root / "datasets"
    tokenizers = root / "tokenizers"
    shard = local_path_for_remote(
        "datasets/datasets/fineweb10B_sp1024/fineweb_train_000000.bin",
        root=root,
        datasets_dir=datasets,
        tokenizers_dir=tokenizers,
        remote_root_prefix="datasets",
    )
    model = local_path_for_remote(
        "datasets/tokenizers/fineweb_1024_bpe.model",
        root=root,
        datasets_dir=datasets,
        tokenizers_dir=tokenizers,
        remote_root_prefix="datasets",
    )
    manifest = local_path_for_remote(
        "datasets/manifest.json",
        root=root,
        datasets_dir=datasets,
        tokenizers_dir=tokenizers,
        remote_root_prefix="datasets",
    )
    assert shard == datasets / "fineweb10B_sp1024" / "fineweb_train_000000.bin"
    assert model == tokenizers / "fineweb_1024_bpe.model"
    assert manifest == root / "manifest.json"


def test_train_shard_count_defaults_and_conflicts():
    assert resolve_train_shard_count(None, None) == 80
    assert resolve_train_shard_count(1, None) == 1
    assert resolve_train_shard_count(None, 10) == 10
    with pytest.raises(ValueError, match="not both"):
        resolve_train_shard_count(1, 10)
    with pytest.raises(ValueError, match="non-negative"):
        resolve_train_shard_count(-1, None)


def test_manifest_shard_counts_require_integers():
    assert manifest_shard_counts({"stats": {"files_train": "80", "files_val": 1}}) == (80, 1)
    with pytest.raises(ValueError, match="files_train"):
        manifest_shard_counts({"stats": {}})


def test_cuda_budget_matches_the_starter_defaults():
    assert cuda_grad_accum_steps(1) == 8
    assert cuda_grad_accum_steps(8) == 1
    assert require_divisible_token_budget(
        train_batch_tokens=524_288,
        world_size=1,
        grad_accum_steps=8,
        seq_len=1024,
    ) == 64
    with pytest.raises(ValueError, match="divisible"):
        require_divisible_token_budget(
            train_batch_tokens=10000,
            world_size=1,
            grad_accum_steps=8,
            seq_len=1024,
        )
    with pytest.raises(ValueError, match="1, 2, 4, 8"):
        cuda_grad_accum_steps(3)
