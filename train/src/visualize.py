from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from core import ParameterNormalizer, evaluate_function, normalize_values


def _build_loss_grid(objective: Callable[[np.ndarray], float], bounds: np.ndarray, grid_size: int = 120):
    a_vals = np.linspace(bounds[0, 0], bounds[0, 1], grid_size)
    b_vals = np.linspace(bounds[1, 0], bounds[1, 1], grid_size)
    aa, bb = np.meshgrid(a_vals, b_vals)

    losses = np.zeros_like(aa)
    for i in range(grid_size):
        for j in range(grid_size):
            losses[i, j] = objective(np.array([aa[i, j], bb[i, j]], dtype=float))

    return aa, bb, losses


def save_loss_landscape_with_path(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    history_params: np.ndarray,
    history_loss: np.ndarray,
    normalizer: ParameterNormalizer,
    out_path: Path,
    title: str,
) -> None:
    aa, bb, losses = _build_loss_grid(objective, bounds)

    params_norm = normalizer.transform_params(history_params)
    grid_points = np.column_stack([aa.ravel(), bb.ravel()])
    grid_points_norm = normalizer.transform_params(grid_points)
    aa_norm = grid_points_norm[:, 0].reshape(aa.shape)
    bb_norm = grid_points_norm[:, 1].reshape(bb.shape)

    losses_norm = normalize_values(losses, normalizer.method)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(aa_norm, bb_norm, losses_norm, levels=30, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="Normalized loss")

    ax.plot(params_norm[:, 0], params_norm[:, 1], color="white", linewidth=1.5, alpha=0.95)
    ax.scatter(
        params_norm[:, 0],
        params_norm[:, 1],
        c=np.arange(len(params_norm)),
        cmap="plasma",
        edgecolors="black",
        s=35,
    )

    ax.set_title(title)
    ax.set_xlabel(f"a ({normalizer.method})")
    ax.set_ylabel(f"b ({normalizer.method})")
    ax.grid(alpha=0.25)

    text = f"best_loss={history_loss[-1]:.6f}\nsteps={len(history_loss)}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left", color="white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_ab_path_animation(
    history_params: np.ndarray,
    bounds: np.ndarray,
    normalizer: ParameterNormalizer,
    out_path: Path,
    title: str,
    fps: int,
) -> None:
    params_norm = normalizer.transform_params(history_params)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(title)
    ax.set_xlabel(f"a ({normalizer.method})")
    ax.set_ylabel(f"b ({normalizer.method})")
    ax.grid(alpha=0.3)

    x_min = float(np.min(params_norm[:, 0]))
    x_max = float(np.max(params_norm[:, 0]))
    y_min = float(np.min(params_norm[:, 1]))
    y_max = float(np.max(params_norm[:, 1]))
    margin = 0.08

    dx = max(x_max - x_min, 1e-6)
    dy = max(y_max - y_min, 1e-6)
    ax.set_xlim(x_min - margin * dx, x_max + margin * dx)
    ax.set_ylim(y_min - margin * dy, y_max + margin * dy)

    line, = ax.plot([], [], color="tab:blue", linewidth=2.0)
    point = ax.scatter([], [], color="tab:red", s=55, edgecolors="black")
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def update(frame: int):
        idx = frame + 1
        x = params_norm[:idx, 0]
        y = params_norm[:idx, 1]
        line.set_data(x, y)
        point.set_offsets(np.array([x[-1], y[-1]]))
        txt.set_text(f"iter={frame}\na={history_params[frame, 0]:.4f}\nb={history_params[frame, 1]:.4f}")
        return line, point, txt

    anim = FuncAnimation(fig, update, frames=len(history_params), interval=int(1000 / max(1, fps)), blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)


def save_y_curve_animation(
    problem: str,
    x: np.ndarray,
    y_obs: np.ndarray,
    y_clean: np.ndarray,
    history_params: np.ndarray,
    out_path: Path,
    title: str,
    fps: int,
) -> None:
    y_preds = [evaluate_function(problem, x, p[0], p[1]) for p in history_params]
    y_all = np.concatenate([y_obs, y_clean] + y_preds)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)

    ax.scatter(x, y_obs, s=12, alpha=0.35, color="gray", label="observed")
    ax.plot(x, y_clean, color="tab:green", linewidth=2.0, label="true function (fixed)")
    pred_line, = ax.plot(x, y_preds[0], color="tab:orange", linewidth=2.0, label="y_pred")
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    pad = 0.08 * max(y_max - y_min, 1e-6)
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.legend(loc="upper right")

    def update(frame: int):
        pred_line.set_ydata(y_preds[frame])
        info.set_text(f"iter={frame}\na={history_params[frame, 0]:.4f}, b={history_params[frame, 1]:.4f}")
        return pred_line, info

    anim = FuncAnimation(fig, update, frames=len(history_params), interval=int(1000 / max(1, fps)), blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(out_path, writer=writer)
    plt.close(fig)
