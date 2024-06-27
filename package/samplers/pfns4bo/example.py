from __future__ import annotations

import warnings

import optuna
import optunahub


module = optunahub.load_module("samplers/pfns4bo")
PFNs4BOSampler = module.PFNs4BOSampler

warnings.filterwarnings("ignore", category=UserWarning)


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    study = optuna.create_study(
        sampler=PFNs4BOSampler(),
    )
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)
