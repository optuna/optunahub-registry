from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


sampler = optunahub.load_module(package="samplers/safe_cma").SafeCMASampler()

study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=30)
print(study.best_trials)
