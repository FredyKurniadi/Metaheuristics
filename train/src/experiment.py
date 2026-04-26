from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from core import ParameterNormalizer, evaluate_function, generate_dataset, mse, read_bounds
from optimizers import run_ga, run_gradient_autograd, run_pso
from visualize import save_ab_path_animation, save_loss_landscape_with_path, save_y_curve_animation


def _resolve_initial_params(problem_cfg: Dict, method_name: str) -> Optional[np.ndarray]:
    init_cfg = problem_cfg.get("initial_params")
    if init_cfg is None:
        return None

    if "a" in init_cfg and "b" in init_cfg:
        return np.array([float(init_cfg["a"]), float(init_cfg["b"])], dtype=float)

    method_cfg = init_cfg.get(method_name)
    if method_cfg is None:
        method_cfg = init_cfg.get("default")
    if method_cfg is None:
        return None

    return np.array([float(method_cfg["a"]), float(method_cfg["b"] )], dtype=float)


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def run_single_experiment(
    cfg: Dict,
    problem_name: str,
    method_name: str,
    output_dir: Optional[Path],
    seed_offset: int,
    optimizer_cfg_override: Optional[Dict] = None,
    save_artifacts: bool = True,
    include_history_loss: bool = False,
) -> Dict:
    rng = np.random.default_rng(int(cfg["seed"]) + seed_offset)

    problem_cfg = cfg["problems"][problem_name]
    bounds = read_bounds(problem_cfg)
    initial_params = _resolve_initial_params(problem_cfg, method_name)

    dataset = generate_dataset(problem_name, cfg, rng)
    x = dataset["x"]
    y_obs = dataset["y_obs"]
    y_clean = dataset["y_clean"]
    true_params = dataset["true_params"]

    def objective(params: np.ndarray) -> float:
        y_pred = evaluate_function(problem_name, x, float(params[0]), float(params[1]))
        return mse(y_obs, y_pred)

    optimizer_cfg = cfg["optimizers"][method_name]
    optimizer_cfg = dict(optimizer_cfg)
    if optimizer_cfg_override:
        optimizer_cfg.update(optimizer_cfg_override)
    if initial_params is not None and "initial_params" not in optimizer_cfg:
        optimizer_cfg["initial_params"] = initial_params

    effective_initial = optimizer_cfg.get("initial_params")
    if effective_initial is not None:
        effective_initial = np.asarray(effective_initial, dtype=float)

    if method_name == "pso":
        result = run_pso(objective, bounds, optimizer_cfg, rng)
    elif method_name == "ga":
        result = run_ga(objective, bounds, optimizer_cfg, rng)
    elif method_name == "gradient_autograd":
        result = run_gradient_autograd(problem_name, x, y_obs, bounds, optimizer_cfg, rng)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    if save_artifacts:
        if output_dir is None:
            raise ValueError("output_dir harus disediakan jika save_artifacts=True")
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_data_points(output_dir / "data_points.csv", x, y_obs, y_clean)
        _save_history(output_dir / "history.csv", result.history_params, result.history_loss)

    vis_norm = cfg["visualization"].get("normalization", "minmax")
    normalizer = ParameterNormalizer(method=vis_norm, bounds=bounds)
    fps = int(cfg["visualization"].get("fps", 12))

    if save_artifacts:
        save_loss_landscape_with_path(
            objective=objective,
            bounds=bounds,
            history_params=result.history_params,
            history_loss=result.history_loss,
            normalizer=normalizer,
            out_path=output_dir / "loss_landscape_path.png",
            title=f"{problem_name} | {method_name.upper()} | Loss Path",
        )

        save_ab_path_animation(
            history_params=result.history_params,
            bounds=bounds,
            normalizer=normalizer,
            out_path=output_dir / "ab_path.gif",
            title=f"{problem_name} | {method_name.upper()} | (a,b) Path",
            fps=fps,
        )

        save_y_curve_animation(
            problem=problem_name,
            x=x,
            y_obs=y_obs,
            y_clean=y_clean,
            history_params=result.history_params,
            out_path=output_dir / "y_pred_vs_true.gif",
            title=f"{problem_name} | {method_name.upper()} | y_pred vs y_true",
            fps=fps,
        )

    summary = {
        "problem": problem_name,
        "method": method_name,
        "true_a": float(true_params[0]),
        "true_b": float(true_params[1]),
        "best_a": float(result.best_params[0]),
        "best_b": float(result.best_params[1]),
        "best_loss": float(result.best_loss),
        "initial_a": float(effective_initial[0]) if effective_initial is not None else None,
        "initial_b": float(effective_initial[1]) if effective_initial is not None else None,
        "steps": int(len(result.history_loss)),
        "population": int(optimizer_cfg["population"]) if "population" in optimizer_cfg else None,
        "iterations": int(optimizer_cfg["iterations"]),
        "optimizer_config": _json_safe(optimizer_cfg),
    }

    if include_history_loss:
        summary["history_loss"] = [float(v) for v in result.history_loss]

    if save_artifacts:
        with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


def _save_data_points(path: Path, x: np.ndarray, y_obs: np.ndarray, y_clean: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y_obs", "y_clean"])
        for xi, yi, yc in zip(x, y_obs, y_clean):
            writer.writerow([float(xi), float(yi), float(yc)])


def _save_history(path: Path, history_params: np.ndarray, history_loss: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "a", "b", "loss"])
        for idx, (params, loss) in enumerate(zip(history_params, history_loss)):
            writer.writerow([idx, float(params[0]), float(params[1]), float(loss)])
