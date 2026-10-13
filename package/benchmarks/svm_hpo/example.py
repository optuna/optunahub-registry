"""Compare TPE and random search on the SVM hyperparameter optimization benchmark.

Each sampler is run with several seeds. Two numbers are reported:

- the mean best cross-validated error, and
- the mean median error of the trials after TPE's 10 random warm-up trials,
  which shows how well each sampler concentrates on good regions.
"""

from __future__ import annotations

import numpy as np
import optuna
import optunahub


optuna.logging.set_verbosity(optuna.logging.WARNING)

svm_hpo = optunahub.load_module("benchmarks/svm_hpo")
problem = svm_hpo.Problem(dataset_name="wine", seed=42)

n_trials = 30
n_warmup = 10  # Default `n_startup_trials` of `TPESampler`.
seeds = [0, 1, 2, 3, 4]


def make_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    if name == "TPE":
        return optuna.samplers.TPESampler(seed=seed)
    return optuna.samplers.RandomSampler(seed=seed)


for name in ["TPE", "Random"]:
    best_errors, median_errors = [], []
    for seed in seeds:
        study = optuna.create_study(
            directions=problem.directions, sampler=make_sampler(name, seed)
        )
        study.optimize(problem, n_trials=n_trials)
        values = [trial.value for trial in study.trials]
        best_errors.append(study.best_value)
        median_errors.append(np.median(values[n_warmup:]))
    print(
        f"{name:>6}: mean best error = {np.mean(best_errors):.4f}, "
        f"mean median error after warm-up = {np.mean(median_errors):.4f}"
    )
