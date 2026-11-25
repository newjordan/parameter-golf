"""Token-budget checks shared by the starter scripts' documented behavior.

``train_gpt.py`` and ``train_gpt_mlx.py`` keep an inline copy so a copied
submission file does not need this package on ``sys.path``.
"""

from __future__ import annotations

VALID_CUDA_WORLD_SIZES = (1, 2, 4, 8)
GRAD_ACCUM_DENOMINATOR = 8


def cuda_grad_accum_steps(world_size: int) -> int:
    if world_size not in VALID_CUDA_WORLD_SIZES:
        raise ValueError(
            f"WORLD_SIZE={world_size} must be one of {', '.join(str(size) for size in VALID_CUDA_WORLD_SIZES)} "
            "so grad_accum_steps stays integral"
        )
    return GRAD_ACCUM_DENOMINATOR // world_size


def require_divisible_token_budget(
    *,
    train_batch_tokens: int,
    world_size: int,
    grad_accum_steps: int,
    seq_len: int,
) -> int:
    if min(train_batch_tokens, world_size, grad_accum_steps, seq_len) <= 0:
        raise ValueError("token budget, world size, grad accumulation, and sequence length must be positive")
    divisor = world_size * grad_accum_steps * seq_len
    if train_batch_tokens % divisor != 0:
        raise ValueError(
            "TRAIN_BATCH_TOKENS must be divisible by WORLD_SIZE * grad_accum_steps * TRAIN_SEQ_LEN "
            f"(got TRAIN_BATCH_TOKENS={train_batch_tokens}, WORLD_SIZE={world_size}, "
            f"grad_accum_steps={grad_accum_steps}, TRAIN_SEQ_LEN={seq_len})"
        )
    return train_batch_tokens // divisor
