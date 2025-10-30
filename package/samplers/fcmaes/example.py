"""
This example is only for sampler.
You can verify your sampler code using this file as well.
Please feel free to remove this file if necessary.
"""

from __future__ import annotations

import numpy as np
import optuna
import optunahub


def SphereIntCOM(x: np.ndarray, z: np.ndarray, c: np.ndarray) -> float:
    return sum(x * x) + sum(z * z) + len(c) - sum(c[:, 0])


def objective(trial: optuna.Trial) -> float:
    x1 = trial.suggest_float("x1", -5, 5)
    x2 = trial.suggest_float("x2", -5, 5)

    z1 = trial.suggest_int("z1", -1, 1)
    z2 = trial.suggest_int("z2", -2, 2)

    c1 = trial.suggest_categorical("c1", [0, 1, 2])
    c2 = trial.suggest_categorical("c2", [0, 1, 2])

    return SphereIntCOM(
        np.array([x1, x2]).reshape(-1, 1),
        np.array([z1, z2]).reshape(-1, 1),
        np.array([c1, c2]).reshape(-1, 1),
    )


module = optunahub.load_module(
    package="samplers/fcmaes"
)


study = optuna.create_study(sampler=module.FastCmaesSampler(16))
study.optimize(objective, n_trials=20)
print(study.best_params)
