from __future__ import annotations

import os

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    z = trial.suggest_float("z", -10, 10)
    return (x - 2) ** 2 + (y + 3) ** 2 + z**2


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.abspath(os.path.join(current_dir, "../../"))

    module = optunahub.load_local_module(
        package="visualization/plot_curved_parallel_coordinate",
        registry_root=package_root,
    )

    fig = module.plot_curved_parallel_coordinate(study)
    fig.show()
