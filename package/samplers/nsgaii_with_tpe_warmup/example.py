from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2


package_name = "samplers/nsgaii_with_tpe_warmup"
sampler = optunahub.load_module(package=package_name).NSGAIIWithTPEWarmupSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=60)
