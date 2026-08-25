"""Minimal working example for the distributionally robust level-set estimation sampler.

The objective depends on a controllable design variable x and an uncontrollable
environmental variable w. The goal is not to optimise it, but to identify the design values
whose distributionally robust probability of exceeding a threshold is above a given level.
"""

import numpy as np
import optuna
import optunahub


x_points = np.linspace(-2.0, 2.0, 24)
w_points = np.linspace(-1.0, 1.0, 10)


def objective(trial: optuna.Trial) -> float:
    x = x_points[trial.suggest_int("x_index", 0, len(x_points) - 1)]
    w = w_points[trial.suggest_int("w_index", 0, len(w_points) - 1)]
    return float(np.sin(2.0 * x) + 0.5 * np.cos(3.0 * w) - 0.3 * (x * w) ** 2)


if __name__ == "__main__":
    sampler = optunahub.load_module("samplers/drlse").DRLevelSetSampler(
        x_points, w_points, h=0.0, alpha=0.4, eps=0.3, seed=0
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=40)

    # The deliverable is the classification, not study.best_trial.
    reliable, unreliable, unclassified = sampler.classify(study)
    print(f"reliable   ({reliable.sum():2d}): {np.round(x_points[reliable], 2)}")
    print(f"unreliable ({unreliable.sum():2d}): {np.round(x_points[unreliable], 2)}")
    print(f"undecided  ({unclassified.sum():2d}): {np.round(x_points[unclassified], 2)}")
