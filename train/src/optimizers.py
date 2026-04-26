from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import autograd.numpy as anp
from autograd import grad
import numpy as np

ObjectiveFn = Callable[[np.ndarray], float]


@dataclass
class OptimResult:
    method: str
    best_params: np.ndarray
    best_loss: float
    history_params: np.ndarray
    history_loss: np.ndarray


def _clip_to_bounds(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    low = bounds[:, 0]
    high = bounds[:, 1]
    return np.clip(values, low, high)


def run_pso(objective: ObjectiveFn, bounds: np.ndarray, cfg: Dict, rng: np.random.Generator) -> OptimResult:
    pop_size = int(cfg["population"])
    iterations = int(cfg["iterations"])
    w = float(cfg.get("w", 0.72))
    c1 = float(cfg.get("c1", 1.6))
    c2 = float(cfg.get("c2", 1.6))

    low = bounds[:, 0]
    high = bounds[:, 1]
    dim = bounds.shape[0]

    init_params = cfg.get("initial_params")
    if init_params is not None:
        base = np.asarray(init_params, dtype=float)
        init_spread = float(cfg.get("init_spread", 0.35))
        particles = base + rng.normal(loc=0.0, scale=init_spread, size=(pop_size, dim))
        particles[0] = base
        particles = _clip_to_bounds(particles, bounds)
    else:
        particles = rng.uniform(low=low, high=high, size=(pop_size, dim))
    velocity = rng.normal(loc=0.0, scale=0.1, size=(pop_size, dim))

    pbest = particles.copy()
    pbest_loss = np.array([objective(p) for p in pbest], dtype=float)
    g_idx = int(np.argmin(pbest_loss))
    gbest = pbest[g_idx].copy()
    gbest_loss = float(pbest_loss[g_idx])

    history_params = [gbest.copy()]
    history_loss = [gbest_loss]

    for _ in range(iterations):
        r1 = rng.uniform(size=(pop_size, dim))
        r2 = rng.uniform(size=(pop_size, dim))
        velocity = (
            w * velocity
            + c1 * r1 * (pbest - particles)
            + c2 * r2 * (gbest.reshape(1, -1) - particles)
        )
        particles = _clip_to_bounds(particles + velocity, bounds)

        losses = np.array([objective(p) for p in particles], dtype=float)
        improved = losses < pbest_loss
        pbest[improved] = particles[improved]
        pbest_loss[improved] = losses[improved]

        g_idx = int(np.argmin(pbest_loss))
        if pbest_loss[g_idx] < gbest_loss:
            gbest = pbest[g_idx].copy()
            gbest_loss = float(pbest_loss[g_idx])

        history_params.append(gbest.copy())
        history_loss.append(gbest_loss)

    return OptimResult(
        method="pso",
        best_params=gbest,
        best_loss=gbest_loss,
        history_params=np.asarray(history_params, dtype=float),
        history_loss=np.asarray(history_loss, dtype=float),
    )


def _tournament_selection(population: np.ndarray, losses: np.ndarray, rng: np.random.Generator, k: int = 3) -> np.ndarray:
    idx = rng.integers(0, len(population), size=k)
    winner = idx[int(np.argmin(losses[idx]))]
    return population[winner]


def run_ga(objective: ObjectiveFn, bounds: np.ndarray, cfg: Dict, rng: np.random.Generator) -> OptimResult:
    pop_size = int(cfg["population"])
    iterations = int(cfg["iterations"])
    elite_ratio = float(cfg.get("elite_ratio", 0.2))
    mutation_rate = float(cfg.get("mutation_rate", 0.1))
    mutation_scale = float(cfg.get("mutation_scale", 0.05))

    low = bounds[:, 0]
    high = bounds[:, 1]
    dim = bounds.shape[0]

    init_params = cfg.get("initial_params")
    if init_params is not None:
        base = np.asarray(init_params, dtype=float)
        init_spread = float(cfg.get("init_spread", 0.45))
        population = base + rng.normal(loc=0.0, scale=init_spread, size=(pop_size, dim))
        population[0] = base
        population = _clip_to_bounds(population, bounds)
    else:
        population = rng.uniform(low=low, high=high, size=(pop_size, dim))

    def evaluate(pop: np.ndarray) -> np.ndarray:
        return np.array([objective(ind) for ind in pop], dtype=float)

    losses = evaluate(population)
    best_idx = int(np.argmin(losses))
    best = population[best_idx].copy()
    best_loss = float(losses[best_idx])

    history_params = [best.copy()]
    history_loss = [best_loss]

    for _ in range(iterations):
        order = np.argsort(losses)
        population = population[order]
        losses = losses[order]

        elite_count = max(1, int(round(elite_ratio * pop_size)))
        next_population = [population[i].copy() for i in range(elite_count)]

        while len(next_population) < pop_size:
            p1 = _tournament_selection(population, losses, rng)
            p2 = _tournament_selection(population, losses, rng)

            alpha = rng.uniform(size=dim)
            child = alpha * p1 + (1.0 - alpha) * p2

            mut_mask = rng.uniform(size=dim) < mutation_rate
            mutation = rng.normal(loc=0.0, scale=mutation_scale, size=dim)
            child = child + mut_mask * mutation
            child = _clip_to_bounds(child, bounds)

            next_population.append(child)

        population = np.asarray(next_population, dtype=float)
        losses = evaluate(population)

        current_idx = int(np.argmin(losses))
        current_best = population[current_idx].copy()
        current_loss = float(losses[current_idx])
        if current_loss < best_loss:
            best = current_best
            best_loss = current_loss

        history_params.append(best.copy())
        history_loss.append(best_loss)

    return OptimResult(
        method="ga",
        best_params=best,
        best_loss=best_loss,
        history_params=np.asarray(history_params, dtype=float),
        history_loss=np.asarray(history_loss, dtype=float),
    )


def run_gradient_autograd(
    problem_name: str,
    x: np.ndarray,
    y_obs: np.ndarray,
    bounds: np.ndarray,
    cfg: Dict,
    rng: np.random.Generator,
) -> OptimResult:
    iterations = int(cfg["iterations"])
    learning_rate = float(cfg.get("learning_rate", 0.05))
    beta1 = float(cfg.get("beta1", 0.9))
    beta2 = float(cfg.get("beta2", 0.999))
    eps = float(cfg.get("epsilon", 1e-8))
    init_params = cfg.get("initial_params")

    low = bounds[:, 0]
    high = bounds[:, 1]
    if init_params is not None:
        theta = _clip_to_bounds(np.asarray(init_params, dtype=float), bounds)
    else:
        theta = rng.uniform(low=low, high=high, size=2).astype(float)

    x_ag = anp.asarray(x, dtype=float)
    y_ag = anp.asarray(y_obs, dtype=float)

    def loss_fn(params: anp.ndarray) -> anp.ndarray:
        a = params[0]
        b = params[1]
        if problem_name == "soal_1":
            pred = anp.sin(a * x_ag) * anp.cos(b * x_ag)
        elif problem_name == "soal_2":
            exp_arg = anp.clip(-a * (x_ag**2), -8.0, 8.0)
            pred = anp.exp(exp_arg) * anp.sin(b * x_ag)
        else:
            raise ValueError(f"Unknown problem: {problem_name}")
        return anp.mean((y_ag - pred) ** 2)

    grad_fn = grad(loss_fn)

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)

    best_params = theta.copy()
    best_loss = float(loss_fn(theta))
    history_params = [best_params.copy()]
    history_loss = [best_loss]

    for t in range(1, iterations + 1):
        g = np.asarray(grad_fn(theta), dtype=float)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)

        theta = theta - learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        theta = _clip_to_bounds(theta, bounds)

        current_loss = float(loss_fn(theta))
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = theta.copy()

        history_params.append(best_params.copy())
        history_loss.append(best_loss)

    return OptimResult(
        method="gradient_autograd",
        best_params=best_params,
        best_loss=best_loss,
        history_params=np.asarray(history_params, dtype=float),
        history_loss=np.asarray(history_loss, dtype=float),
    )
