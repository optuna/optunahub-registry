from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    model = trial.suggest_categorical("model", ["linear", "tree"])
    alpha = trial.suggest_float("alpha", 1e-4, 1.0, log=True)

    if model == "tree":
        max_depth = trial.suggest_int("max_depth", 2, 10)
        return alpha, max_depth / 10

    return alpha, 0.5


def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    depth = trial.params.get("max_depth", 5) / 10
    return (trial.params["alpha"] + depth - 0.7,)


study = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=optuna.samplers.NSGAIISampler(constraints_func=constraints, seed=0),
)
study.optimize(objective, n_trials=30)

module = optunahub.load_module(package="visualization/extended_pcp")
fig = module.plot_parallel_coordinate(study)
fig.show()
