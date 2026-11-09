from __future__ import annotations

import numpy as np
import optuna
import optunahub


# Binary quadratic program from Sect. 4.1 of Baptista & Poloczek (2018): maximize x^T Q x
# over x in {0, 1}^d, where Q has decaying correlations between nearby variables.
d = 10
rng = np.random.default_rng(0)
idx = np.arange(d)
Q = rng.normal(size=(d, d)) * np.exp(-((idx[:, None] - idx[None, :]) ** 2) / 10.0**2)


def objective(trial: optuna.Trial) -> float:
    x = np.array([trial.suggest_int(f"x{i}", 0, 1) for i in range(d)])
    return float(x @ Q @ x)


module = optunahub.load_module("samplers/bocs")
for acquisition in ("sdp", "sa"):
    sampler = module.BOCSSampler(acquisition=acquisition, seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=50)
    print(f"BOCS-{acquisition.upper()}: best value {study.best_value:.4f}")
