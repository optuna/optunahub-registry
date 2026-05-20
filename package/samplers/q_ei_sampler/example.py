from __future__ import annotations

import optuna
import optunahub


module = optunahub.load_module(package="samplers/q_ei_sampler")
DimSpec = module.DimSpec
qEISampler = module.qEISampler

sampler = qEISampler(
    search_space=[DimSpec("x", "float", -5.0, 5.0)],
    api_url="https://your-gp-service/suggest",  # substitute your own endpoint
)

study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(lambda trial: trial.suggest_float("x", -5, 5) ** 2, n_trials=20)
print("Best value:", study.best_value)
