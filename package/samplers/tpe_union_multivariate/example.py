from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_categorical("x", [True, False])
    y = trial.suggest_float("y", -1, 1)
    if x:
        a = trial.suggest_float("a", -1, 1)
        return (a - y) ** 2
    else:
        b = trial.suggest_float("b", -1, 1)
        return (b - y) ** 2


if __name__ == "__main__":
    module = optunahub.load_module(package="samplers/tpe_union_multivariate")
    sampler = module.TPEUnionMultivariateSampler(n_startup_trials=5, seed=42)

    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=20)
    print(f"Best trial value: {study.best_value}")
