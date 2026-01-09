from __future__ import annotations

from typing import Sequence

import optuna
import optunahub


def objective_function(x: Sequence[float]) -> float:
    return sum(x[i] ** 2 for i in range(len(x)))


def safe_function(x: Sequence[float]) -> float:
    return x[0]


def objective(trial: optuna.Trial) -> float:
    x1 = trial.suggest_float("x1", -5, 5)
    x2 = trial.suggest_float("x2", -5, 5)
    return objective_function([x1, x2])


safe_seeds = [[-2.0, -2.0], [-1.0, 1.0], [-0.5, 3.0]]
seeds_evals = [objective_function(x) for x in safe_seeds]
seeds_safe_evals = [[safe_function(x)] for x in safe_seeds]
safety_threshold = [0.0]

sampler = optunahub.load_module(
    package="samplers/safe_cma",
).SafeCMASampler(
    safe_seeds=safe_seeds,
    seeds_evals=seeds_evals,
    seeds_safe_evals=seeds_safe_evals,
    safety_threshold=safety_threshold,
    safe_function=safe_function,
    seed=42,
)

study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=500)

fig = optuna.visualization.plot_optimization_history(study)
fig.show()
