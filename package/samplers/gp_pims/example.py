from __future__ import annotations

import numpy as np
import optuna
import optunahub


def f(x):
    return -np.sin(3 * np.sum(x**2)) - np.sum(x**2) ** 2 + 0.7 * np.sum(x**2)


if __name__ == "__main__":
    mod = optunahub.load_module(
        package="samplers/gp_pims",
    )

    PIMSSampler = mod.PIMSSampler

    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        y = trial.suggest_float("y", 0, 1)
        return f(np.asarray([x, y]))

    search_space = {
        "x": optuna.distributions.FloatDistribution(0, 1),
        "y": optuna.distributions.FloatDistribution(0, 1),
    }

    kernel_bounds = np.array([[1e-3, 1e-3], [1e3, 1e3]])
    sampler = PIMSSampler(search_space=search_space, kernel_bounds=kernel_bounds)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    # study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)
    # optuna.visualization.plot_optimization_history(study)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image("optuna_history.png")
