from __future__ import annotations

import optuna
import optunahub


def objective(t: optuna.Trial) -> tuple[float, float]:
    x = t.suggest_float("x", -5, 5)
    y = t.suggest_float("y", -5, 5)
    return (x + 2)**2 + (y + 2)**2, (x - 2)**2 + (y - 2)**2


sampler = optuna.samplers.TPESampler(seed=0)
study = optuna.create_study(sampler=sampler, directions=["minimize"]*2)
study.optimize(objective, n_trials=200)
get_hypervolume_history = optunahub.load_module(
    "benchmarks/hypervolume_history"
).get_hypervolume_history
print(get_hypervolume_history(study))
