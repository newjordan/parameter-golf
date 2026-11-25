import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cuda_and_mlx_help_exit_before_optional_imports():
    cuda = _run("train_gpt.py", "--help")
    mlx = _run("train_gpt_mlx.py", "--help")
    assert cuda.returncode == 0
    assert mlx.returncode == 0
    assert "16000000" in cuda.stdout
    assert "TRAIN_BATCH_TOKENS must be divisible" in cuda.stdout
    assert "CUDA" in cuda.stdout
    assert "ModuleNotFoundError" not in cuda.stderr
    assert "mlx" in mlx.stdout.lower()
    assert "VAL_BATCH_SIZE / GRAD_ACCUM_STEPS" in mlx.stdout
    assert "ModuleNotFoundError" not in mlx.stderr


def test_unknown_training_arguments_are_rejected():
    cuda = _run("train_gpt.py", "--epochs", "1")
    mlx = _run("train_gpt_mlx.py", "train")
    assert cuda.returncode == 2
    assert mlx.returncode == 2
    assert "Unrecognized arguments" in cuda.stderr
    assert "Unrecognized arguments" in mlx.stderr


def test_starters_keep_the_budget_guard_inline():
    cuda = (ROOT / "train_gpt.py").read_text(encoding="utf-8")
    mlx = (ROOT / "train_gpt_mlx.py").read_text(encoding="utf-8")
    assert "TRAIN_BATCH_TOKENS must be divisible by WORLD_SIZE * grad_accum_steps * TRAIN_SEQ_LEN" in cuda
    assert "TRAIN_BATCH_TOKENS must be divisible by WORLD_SIZE * grad_accum_steps * TRAIN_SEQ_LEN" in mlx
    assert len(cuda.splitlines()) < 1500
    assert len(mlx.splitlines()) < 1500
