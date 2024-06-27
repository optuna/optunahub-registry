from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.distributions import FloatDistribution
import optunahub


PLMBOSampler = optunahub.load_module(  # type: ignore
    "samplers/plmbo",
).PLMBOSampler

if __name__ == "__main__":
    f_sigma = 0.01

    def obj_func1(x):
        return np.sin(x[0]) + x[1]

    def obj_func2(x):
        return -np.sin(x[0]) - x[1] + 0.1

    def obs_obj_func(x):
        return np.array(
            [
                obj_func1(x) + np.random.normal(0, f_sigma),
                obj_func2(x) + np.random.normal(0, f_sigma),
            ]
        )

    def objective(trial: optuna.Trial):
        x1 = trial.suggest_float("x1", 0, 1)
        x2 = trial.suggest_float("x2", 0, 1)
        values = obs_obj_func(np.array([x1, x2]))
        return float(values[0]), float(values[1])

    sampler = PLMBOSampler(
        {
            "x1": FloatDistribution(0, 1),
            "x2": FloatDistribution(0, 1),
        }
    )
    study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=20)

    optuna.visualization.matplotlib.plot_pareto_front(study)
    plt.show()
