from __future__ import annotations

import matplotlib.pyplot as plt
import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2


multiple_study_list = []
for sampler_cls in [optuna.samplers.RandomSampler, optuna.samplers.TPESampler]:
    study_list = []
    for seed in range(10):
        sampler = sampler_cls(seed=seed)
        study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
        study.optimize(objective, n_trials=50)
        study_list.append(study)
    multiple_study_list.append(study_list)

plot_multiple_empirical_attainment_surfaces = optunahub.load_module(
    "package/visualization/plot_empirical_attainment_surface"
).plot_multiple_empirical_attainment_surfaces
ax = plot_multiple_empirical_attainment_surfaces(
    multiple_study_list, levels=[3, 5, 7], labels=["Random", "TPE"]
)
plt.show()
