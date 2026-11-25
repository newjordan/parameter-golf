"""Helpers for the Parameter Golf starter kit and research hub.

Training entry points stay self-contained. This package holds the pieces that
can be tested without CUDA, MLX, model weights, or a dataset download.
"""

ARTIFACT_CAP_BYTES = 16_000_000

__all__ = ["ARTIFACT_CAP_BYTES"]
