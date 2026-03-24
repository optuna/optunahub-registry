"""
Example: CmaEsRefinementSampler vs plain CmaEsSampler on Rastrigin (5D).

Demonstrates the benefit of the refinement phase on a multimodal function.
Requires cmaes and scipy (see requirements.txt).
"""

from __future__ import annotations

import math

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    """5D Rastrigin function — multimodal with many local optima."""
    n = 5
    variables = [trial.suggest_float(f"x{i}", -5.12, 5.12) for i in range(n)]
    A = 10
    return A * n + sum(x**2 - A * math.cos(2 * math.pi * x) for x in variables)


def run_comparison(n_trials: int = 200, n_seeds: int = 5) -> None:
    """Compare CmaEsRefinementSampler against plain CmaEsSampler."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    module = optunahub.load_module("samplers/cma_es_refinement")

    refinement_values = []
    cmaes_values = []

    for seed in range(n_seeds):
        # CMA-ES + Refinement
        sampler = module.CmaEsRefinementSampler(seed=seed)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=n_trials)
        refinement_values.append(study.best_value)

        # Plain CMA-ES (same budget)
        sampler_plain = optuna.samplers.CmaEsSampler(
            seed=seed, sigma0=0.2, popsize=6, n_startup_trials=8
        )
        study_plain = optuna.create_study(sampler=sampler_plain)
        study_plain.optimize(objective, n_trials=n_trials)
        cmaes_values.append(study_plain.best_value)

    ref_mean = sum(refinement_values) / len(refinement_values)
    cma_mean = sum(cmaes_values) / len(cmaes_values)

    print(f"Rastrigin 5D — {n_trials} trials × {n_seeds} seeds")
    print(f"  CMA-ES + Refinement: {ref_mean:.4f} (mean best value)")
    print(f"  Plain CMA-ES:        {cma_mean:.4f} (mean best value)")
    print(f"  Improvement:         {(cma_mean - ref_mean) / cma_mean * 100:.1f}%")


if __name__ == "__main__":
    run_comparison()
