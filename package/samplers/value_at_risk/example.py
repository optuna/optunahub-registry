from __future__ import annotations

import numpy as np
import optuna
import optunahub


def numpy_objective(X: np.ndarray) -> np.ndarray:
    C = np.asarray([[2.0, -12.2, 21.2, -6.4, -4.7, 6.2], [1.0, -11, 43.3, -74.8, 56.9, -10]])
    X_poly = np.zeros_like(X)
    for i in range(C.shape[1]):
        X_poly = X * (X_poly + C[:, i])
    X0X1 = X[..., 0] * X[..., 1]
    out = np.sum(X_poly, axis=-1)
    out += X0X1 * (-4.1 - 0.1 * X0X1 + 0.4 * X[..., 0] + 0.4 * X[..., 1])
    return out


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -1, 4)
    y = trial.suggest_float("y", -1, 4)
    X = np.array([x, y])
    f = numpy_objective(X).item()
    return f


RobustGPSampler = optunahub.load_module("samplers/value_at_risk").RobustGPSampler
sampler = RobustGPSampler(seed=0, uniform_input_noise_ranges={"x": 0.5, "y": 0.5})
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=50)
