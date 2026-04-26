from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from core import evaluate_function, generate_dataset, mse


def test_mse_zero_when_equal() -> None:
    y = np.array([1.0, 2.0, 3.0], dtype=float)
    assert mse(y, y) == 0.0


def test_generate_dataset_without_noise_matches_clean() -> None:
    cfg = {
        "data": {
            "num_samples": 25,
            "x_range": [-2.0, 2.0],
            "noise": {"type": "gaussian", "params": {"mean": 0.0, "std": 0.0}},
        },
        "problems": {
            "soal_1": {
                "true_params": {"a": 1.2, "b": 0.9},
                "bounds": {"a": [0.0, 2.0], "b": [0.0, 2.0]},
            }
        },
    }
    rng = np.random.default_rng(123)
    data = generate_dataset("soal_1", cfg, rng)
    np.testing.assert_allclose(data["y_obs"], data["y_clean"], atol=1e-12)


def test_evaluate_function_shape() -> None:
    x = np.linspace(-1.0, 1.0, 11)
    y1 = evaluate_function("soal_1", x, 1.0, 1.0)
    y2 = evaluate_function("soal_2", x, 1.0, 1.0)
    assert y1.shape == x.shape
    assert y2.shape == x.shape
