from __future__ import annotations

import matplotlib.pyplot as plt
import optuna
import optunahub


WhaleOptimizationSampler = optunahub.load_module(  # type: ignore
    "samplers/whale_optimization"
).WhaleOptimizationSampler

if __name__ == "__main__":

    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y**2

    sampler = WhaleOptimizationSampler(
        {
            "x": optuna.distributions.FloatDistribution(-10, 10),
            "y": optuna.distributions.FloatDistribution(-10, 10),
        }
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.show()
