from __future__ import annotations

import optuna
import optunahub
from quantecarlo import fantasize_suggest


module = optunahub.load_module(package="samplers/batch_sampler")
DimSpec = module.DimSpec
BatchSampler = module.BatchSampler

search_space = [
    DimSpec("x", "float", -5.0, 5.0),
    DimSpec("y", "float", -5.0, 5.0),
]

sampler = BatchSampler(
    search_space=search_space,
    suggest_fn=fantasize_suggest,
    q=4,
    n_startup_trials=8,
)

study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(
    lambda trial: trial.suggest_float("x", -5.0, 5.0) ** 2
    + trial.suggest_float("y", -5.0, 5.0) ** 2,
    n_trials=32,
    n_jobs=4,
)
print("Best value:", study.best_value)
print("Best params:", study.best_trial.params)
