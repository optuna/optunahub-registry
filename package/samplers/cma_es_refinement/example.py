"""
Minimal example demonstrating CmaEsRefinementSampler on the Sphere function.

No additional packages beyond optuna and optunahub are required.
"""

from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    """5D Sphere function."""
    variables = [trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(5)]
    return sum(x**2 for x in variables)


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    module = optunahub.load_module("samplers/cma_es_refinement")
    sampler = module.CmaEsRefinementSampler(seed=42)

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=200)

    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
