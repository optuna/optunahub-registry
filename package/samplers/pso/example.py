from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return x**2 + y**2


package_name = "samplers/pso"
test_local = False

n_trials = 100
n_generations = 5

if test_local:
    # This is an example of how to load a sampler from your local optunahub-registry.
    sampler = optunahub.load_local_module(
        package=package_name,
        registry_root="./package",  # Path to the root of the optunahub-registry.
    ).PSOSampler(
        {
            "x": optuna.distributions.FloatDistribution(-10, 10),
            "y": optuna.distributions.FloatDistribution(-10, 10),
        },
        n_particles=int(n_trials / n_generations),
        inertia=0.5,
        cognitive=1.5,
        social=1.5,
    )
else:
    # This is an example of how to load a sampler from your fork of the optunahub-registry.
    sampler = optunahub.load_module(package=package_name).PSOSampler(
        {
            "x": optuna.distributions.FloatDistribution(-10, 10),
            "y": optuna.distributions.FloatDistribution(-10, 10, step=0.1),
        },
        n_particles=int(n_trials / n_generations),
        inertia=0.5,
        cognitive=1.5,
        social=1.5,
    )

study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=n_trials)
print(study.best_trials)
