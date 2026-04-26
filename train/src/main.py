from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import optuna
import yaml

from experiment import run_single_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run METAHEURISTIK experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="train/configs/experiment.yaml",
        help="Path ke file konfigurasi YAML",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def next_model_dir(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted([p.name for p in models_dir.iterdir() if p.is_dir() and p.name.startswith("model_")])
    if not existing:
        return models_dir / "model_001"
    latest_num = max(int(name.split("_")[1]) for name in existing)
    return models_dir / f"model_{latest_num + 1:03d}"


def _sample_optimizer_cfg(method_name: str, base_cfg: Dict, search_space: Dict, trial: optuna.Trial) -> Dict:
    cfg = dict(base_cfg)
    method_space = search_space.get(method_name, {})

    for key, spec in method_space.items():
        if not isinstance(spec, dict):
            continue

        p_type = str(spec.get("type", "float")).lower()
        if p_type == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            cfg[key] = trial.suggest_int(key, low, high, step=step)
        elif p_type == "categorical":
            cfg[key] = trial.suggest_categorical(key, spec["choices"])
        else:
            low = float(spec["low"])
            high = float(spec["high"])
            log = bool(spec.get("log", False))
            step = spec.get("step")
            if step is None:
                cfg[key] = trial.suggest_float(key, low, high, log=log)
            else:
                cfg[key] = trial.suggest_float(key, low, high, step=float(step), log=log)

    return cfg


def _find_convergence_step(history_loss: list[float], selection_cfg: Dict) -> Optional[int]:
    if not history_loss:
        return None

    threshold = float(selection_cfg.get("convergence_loss_threshold", 0.02))
    patience = int(selection_cfg.get("convergence_patience", 10))
    delta_tol = float(selection_cfg.get("convergence_delta_tol", 1e-5))

    n = len(history_loss)
    if n <= patience:
        return n - 1 if history_loss[-1] <= threshold else None

    for i in range(0, n - patience):
        if history_loss[i] <= threshold and (history_loss[i] - history_loss[i + patience]) <= delta_tol:
            return i
    return None


def _select_initial_for_visualization(
    cfg: Dict,
    problem_name: str,
    method_name: str,
    seed_offset: int,
    best_cfg_override: Dict,
    selection_cfg: Dict,
) -> Dict:
    problem_cfg = cfg["problems"][problem_name]
    candidate_inits = problem_cfg.get("evaluation_initial_params", [])
    if not candidate_inits:
        raise ValueError(f"evaluation_initial_params belum diatur untuk {problem_name}")

    runs = []
    for idx, init in enumerate(candidate_inits):
        init_params = [float(init["a"]), float(init["b"])]
        run_cfg_override = dict(best_cfg_override)
        run_cfg_override["initial_params"] = init_params

        summary = run_single_experiment(
            cfg=cfg,
            problem_name=problem_name,
            method_name=method_name,
            output_dir=None,
            seed_offset=seed_offset,
            optimizer_cfg_override=run_cfg_override,
            save_artifacts=False,
            include_history_loss=True,
        )
        conv_step = _find_convergence_step(summary.get("history_loss", []), selection_cfg)
        summary["candidate_index"] = idx
        summary["convergence_step"] = conv_step
        summary["is_converged"] = conv_step is not None
        runs.append(summary)

    converged = [r for r in runs if r["is_converged"]]
    if converged:
        # Choose the converged run that takes the longest steps to converge.
        selected = max(converged, key=lambda r: int(r["convergence_step"]))
    else:
        selected = min(runs, key=lambda r: float(r["best_loss"]))

    return {
        "selected": selected,
        "all_runs": runs,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)

    models_dir = Path(cfg["output"].get("models_dir", "models"))
    run_dir = next_model_dir(models_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    tuning_cfg = cfg.get("tuning", {})
    tuning_enabled = bool(tuning_cfg.get("enabled", False))
    n_trials = int(tuning_cfg.get("n_trials", 12))
    n_trials_by_problem = tuning_cfg.get("n_trials_by_problem", {})
    n_trials_by_problem_method = tuning_cfg.get("n_trials_by_problem_method", {})
    timeout_sec = tuning_cfg.get("timeout_sec")
    search_space = tuning_cfg.get("search_space", {})
    seed_stride = int(tuning_cfg.get("seed_stride", 100))
    optuna_sampler_seed = int(cfg.get("seed", 42)) + int(tuning_cfg.get("sampler_seed_offset", 1000))
    selection_cfg = cfg.get("selection", {})

    for p_idx, problem_name in enumerate(cfg["problems"].keys()):
        problem_seed_offset = p_idx * seed_stride
        problem_n_trials = int(n_trials_by_problem.get(problem_name, n_trials))
        for method_name in cfg["optimizers"].keys():
            out_dir = run_dir / problem_name / method_name
            method_override = n_trials_by_problem_method.get(problem_name, {}).get(method_name)
            effective_n_trials = int(method_override) if method_override is not None else problem_n_trials

            best_cfg_override = None
            study_summary = None

            if tuning_enabled:
                base_method_cfg = dict(cfg["optimizers"][method_name])
                tuning_init = cfg["problems"][problem_name].get("tuning_initial_params")
                if tuning_init is not None:
                    base_method_cfg["initial_params"] = [float(tuning_init["a"]), float(tuning_init["b"])]

                sampler = optuna.samplers.TPESampler(seed=optuna_sampler_seed + p_idx)
                study = optuna.create_study(direction="minimize", sampler=sampler)

                def objective(trial: optuna.Trial) -> float:
                    sampled_cfg = _sample_optimizer_cfg(method_name, base_method_cfg, search_space, trial)
                    trial_summary = run_single_experiment(
                        cfg=cfg,
                        problem_name=problem_name,
                        method_name=method_name,
                        output_dir=None,
                        seed_offset=problem_seed_offset,
                        optimizer_cfg_override=sampled_cfg,
                        save_artifacts=False,
                    )
                    return float(trial_summary["best_loss"])

                study.optimize(objective, n_trials=effective_n_trials, timeout=timeout_sec)
                best_cfg_override = _sample_optimizer_cfg(
                    method_name,
                    base_method_cfg,
                    search_space,
                    optuna.trial.FixedTrial(study.best_params),
                )
                study_summary = {
                    "best_value": float(study.best_value),
                    "best_params": study.best_params,
                    "n_trials": len(study.trials),
                    "configured_n_trials": effective_n_trials,
                }

                out_dir.mkdir(parents=True, exist_ok=True)
                with (out_dir / "optuna_best.json").open("w", encoding="utf-8") as f:
                    json.dump(study_summary, f, indent=2)

                init_selection = _select_initial_for_visualization(
                    cfg=cfg,
                    problem_name=problem_name,
                    method_name=method_name,
                    seed_offset=problem_seed_offset,
                    best_cfg_override=best_cfg_override,
                    selection_cfg=selection_cfg,
                )
                selected_init = init_selection["selected"]
                best_cfg_override["initial_params"] = [
                    float(selected_init["initial_a"]),
                    float(selected_init["initial_b"]),
                ]

                with (out_dir / "initial_selection.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "criterion": "longest_converged_if_any_else_best_loss",
                            "selection_config": selection_cfg,
                            "selected_candidate": {
                                "candidate_index": selected_init["candidate_index"],
                                "initial_a": selected_init["initial_a"],
                                "initial_b": selected_init["initial_b"],
                                "is_converged": selected_init["is_converged"],
                                "convergence_step": selected_init["convergence_step"],
                                "best_loss": selected_init["best_loss"],
                            },
                            "all_candidates": [
                                {
                                    "candidate_index": r["candidate_index"],
                                    "initial_a": r["initial_a"],
                                    "initial_b": r["initial_b"],
                                    "is_converged": r["is_converged"],
                                    "convergence_step": r["convergence_step"],
                                    "best_loss": r["best_loss"],
                                }
                                for r in init_selection["all_runs"]
                            ],
                        },
                        f,
                        indent=2,
                    )

            summary = run_single_experiment(
                cfg=cfg,
                problem_name=problem_name,
                method_name=method_name,
                output_dir=out_dir,
                seed_offset=problem_seed_offset,
                optimizer_cfg_override=best_cfg_override,
                save_artifacts=True,
            )

            if study_summary is not None:
                summary["optuna"] = study_summary
                summary["selection_policy"] = "longest_converged_if_any_else_best_loss"

            summaries.append(summary)
            print(f"[done] {problem_name} | {method_name} | best_loss={summary['best_loss']:.6f}")

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"results": summaries}, f, indent=2)

    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[output] {run_dir}")


if __name__ == "__main__":
    main()
