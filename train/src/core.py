from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Literal

import numpy as np

ProblemName = Literal["soal_1", "soal_2"]


@dataclass
class ParameterNormalizer:
    method: str
    bounds: np.ndarray

    def transform_params(self, params: np.ndarray) -> np.ndarray:
        params = np.asarray(params, dtype=float)
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        eps = 1e-12
        if self.method == "minmax":
            return (params - low) / (high - low + eps)
        if self.method == "zscore":
            center = 0.5 * (low + high)
            scale = 0.5 * (high - low) + eps
            return (params - center) / scale
        return params


def normalize_values(values: np.ndarray, method: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    eps = 1e-12
    if method == "minmax":
        vmin = values.min()
        vmax = values.max()
        return (values - vmin) / (vmax - vmin + eps)
    if method == "zscore":
        return (values - values.mean()) / (values.std() + eps)
    return values


def evaluate_function(problem: ProblemName, x: np.ndarray, a: float, b: float) -> np.ndarray:
    if problem == "soal_1":
        return np.sin(a * x) * np.cos(b * x)
    if problem == "soal_2":
        exp_arg = np.clip(-a * (x**2), -8.0, 8.0)
        return np.exp(exp_arg) * np.sin(b * x)
    raise ValueError(f"Unknown problem: {problem}")


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float(np.mean(diff * diff))


def make_noise(size: int, noise_cfg: Dict, rng: np.random.Generator) -> np.ndarray:
    noise_type = noise_cfg.get("type", "gaussian")
    params = noise_cfg.get("params", {})

    if noise_type == "gaussian":
        mean = float(params.get("mean", 0.0))
        std = float(params.get("std", 0.05))
        return rng.normal(loc=mean, scale=std, size=size)

    if noise_type == "uniform":
        low = float(params.get("low", -0.05))
        high = float(params.get("high", 0.05))
        return rng.uniform(low=low, high=high, size=size)

    raise ValueError(f"Unsupported noise type: {noise_type}")


def _compute_soal1_one_period(a: float, b: float, max_denominator: int = 120) -> float:
    omega_sum = abs(a + b)
    omega_diff = abs(a - b)

    eps = 1e-12
    if omega_sum < eps and omega_diff < eps:
        return 2.0 * np.pi

    if omega_diff < eps:
        return 2.0 * np.pi / max(omega_sum, eps)

    t_sum = 2.0 * np.pi / omega_sum
    t_diff = 2.0 * np.pi / omega_diff

    ratio = omega_sum / omega_diff
    frac = Fraction(float(ratio)).limit_denominator(max_denominator)

    t_candidate_1 = frac.numerator * t_sum
    t_candidate_2 = frac.denominator * t_diff
    if abs(t_candidate_1 - t_candidate_2) <= 1e-6 * max(1.0, t_candidate_1, t_candidate_2):
        return float(0.5 * (t_candidate_1 + t_candidate_2))

    # Fallback when frequencies are effectively incommensurate: show one full slow-beat cycle.
    return float(max(t_sum, t_diff))


def _resolve_x_range(problem: ProblemName, cfg: Dict, true_a: float, true_b: float) -> tuple[float, float]:
    data_cfg = cfg["data"]

    x_range_default = data_cfg.get("x_range_default", data_cfg.get("x_range", [-6.0, 6.0]))
    x_min, x_max = float(x_range_default[0]), float(x_range_default[1])

    if problem != "soal_1":
        return x_min, x_max

    p1_mode = str(data_cfg.get("soal_1_x_mode", "fixed")).lower()
    if p1_mode != "one_period":
        return x_min, x_max

    x_start = float(data_cfg.get("soal_1_x_start", 0.0))
    max_den = int(data_cfg.get("soal_1_period_max_denominator", 120))
    period = _compute_soal1_one_period(true_a, true_b, max_denominator=max_den)
    return x_start, x_start + period


def generate_dataset(problem: ProblemName, cfg: Dict, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    num_samples = int(cfg["data"]["num_samples"])

    true_a = float(cfg["problems"][problem]["true_params"]["a"])
    true_b = float(cfg["problems"][problem]["true_params"]["b"])

    x_min, x_max = _resolve_x_range(problem, cfg, true_a, true_b)

    x = np.linspace(float(x_min), float(x_max), num_samples)
    y_clean = evaluate_function(problem, x, true_a, true_b)
    noise = make_noise(num_samples, cfg["data"]["noise"], rng)
    y_obs = y_clean + noise

    return {
        "x": x,
        "y_clean": y_clean,
        "y_obs": y_obs,
        "true_params": np.array([true_a, true_b], dtype=float),
    }


def read_bounds(problem_cfg: Dict) -> np.ndarray:
    a_low, a_high = problem_cfg["bounds"]["a"]
    b_low, b_high = problem_cfg["bounds"]["b"]
    return np.array([[float(a_low), float(a_high)], [float(b_low), float(b_high)]], dtype=float)
