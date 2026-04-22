from __future__ import annotations

import matplotlib.pyplot as plt
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


plot_target_over_time = optunahub.load_module("visualization/plot_target_over_time").plot_target_over_time
_, ax = plt.subplots()
colors = ["darkred", "black"]
for sampler, color in zip([optuna.samplers.TPESampler(), optuna.samplers.RandomSampler()], colors):
    study_list = []
    for __ in range(5):
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=20)
        study_list.append(study)
    plot_target_over_time(
        study_list,
        ax=ax,
        color=color,
        label=sampler.__class__.__name__,
    )

ax.legend()
plt.show()
