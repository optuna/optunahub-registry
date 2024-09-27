from __future__ import annotations

import matplotlib.pyplot as plt
import optuna
import optunahub


GreyWolfOptimizationSampler = optunahub.load_module(  # type: ignore
    "samplers/grey_wolf_optimization"
).GreyWolfOptimizationSampler


if __name__ == "__main__":

    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y**2

    n_trials = 100

    sampler = GreyWolfOptimizationSampler(
        {
            "x": optuna.distributions.FloatDistribution(-10, 10),
            "y": optuna.distributions.FloatDistribution(-10, 10),
        },
        max_iter=n_trials,  # This should be equal to `n_trials` in `study.optimize`.
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.show()
