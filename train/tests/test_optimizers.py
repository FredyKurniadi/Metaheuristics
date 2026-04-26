from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from core import generate_dataset, read_bounds
from optimizers import run_ga, run_gradient_autograd, run_pso


def test_pso_finds_quadratic_minimum() -> None:
    bounds = np.array([[0.0, 2.0], [0.0, 2.0]], dtype=float)
    target = np.array([1.3, 0.7], dtype=float)

    def objective(p: np.ndarray) -> float:
        return float(np.sum((p - target) ** 2))

    cfg = {"population": 30, "iterations": 60, "w": 0.72, "c1": 1.6, "c2": 1.6}
    rng = np.random.default_rng(42)

    result = run_pso(objective, bounds, cfg, rng)
    assert result.best_loss < 1e-2


def test_ga_finds_quadratic_minimum() -> None:
    bounds = np.array([[0.0, 2.0], [0.0, 2.0]], dtype=float)
    target = np.array([1.1, 1.6], dtype=float)

    def objective(p: np.ndarray) -> float:
        return float(np.sum((p - target) ** 2))

    cfg = {
        "population": 40,
        "iterations": 80,
        "elite_ratio": 0.2,
        "mutation_rate": 0.2,
        "mutation_scale": 0.08,
    }
    rng = np.random.default_rng(123)

    result = run_ga(objective, bounds, cfg, rng)
    assert result.best_loss < 5e-2


def test_gradient_autograd_improves_on_noise_free_problem() -> None:
    cfg = {
        "data": {
            "num_samples": 220,
            "x_range": [-6.0, 6.0],
            "noise": {"type": "gaussian", "params": {"mean": 0.0, "std": 0.0}},
        },
        "problems": {
            "soal_1": {
                "true_params": {"a": 1.25, "b": 0.85},
                "bounds": {"a": [0.0, 2.0], "b": [0.0, 2.0]},
            }
        },
    }
    rng = np.random.default_rng(10)
    data = generate_dataset("soal_1", cfg, rng)
    bounds = read_bounds(cfg["problems"]["soal_1"])

    result = run_gradient_autograd(
        problem_name="soal_1",
        x=data["x"],
        y_obs=data["y_obs"],
        bounds=bounds,
        cfg={
            "iterations": 260,
            "learning_rate": 0.02,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "initial_params": [0.1, 1.9],
        },
        rng=np.random.default_rng(55),
    )

    assert result.best_loss < result.history_loss[0]
    assert result.best_loss < 0.2
