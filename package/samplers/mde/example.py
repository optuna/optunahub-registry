from __future__ import annotations

import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    # 10-dimensional Rastrigin function (global minimum 0 at the origin).
    x = np.array([trial.suggest_float(f"x{i}", -5.12, 5.12) for i in range(10)])
    return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


package_name = "samplers/mde"
test_local = False

if test_local:
    # This is an example of how to load a sampler from your local optunahub-registry.
    module = optunahub.load_local_module(
        package=package_name,
        registry_root="./package",  # Path to the root of the optunahub-registry.
    )
else:
    module = optunahub.load_module(package=package_name)

# The full MDE population has 3 * population_size members. The first
# 3 * population_size trials evaluate the initial population, every following
# generation consists of population_size trials.
sampler = module.MDESampler(population_size=30, seed=42)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=3000)
print(study.best_value, study.best_params)
