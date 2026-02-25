"""
Example demonstrating HeteroscedasticGPSampler.

This Gaussian process-based sampler natively supports heteroscedastic
(input-dependent) observation noise. It is useful when evaluation noise varies
across the parameter space and you have an analytical expression or empirical
estimate of the noise variance for each trial.
"""

import math

import numpy as np
import optuna
import optunahub


optuna.logging.set_verbosity(optuna.logging.WARNING)

module = optunahub.load_module(package="samplers/heteroscedastic_gp")

HeteroscedasticGPSampler = module.HeteroscedasticGPSampler


def branin(x1: float, x2: float) -> float:
    # Branin-Hoo function
    a, b, c = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * math.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s


def noisy_branin_objective(trial: optuna.Trial) -> float:
    # Single-objective Branin with spatially-varying Gaussian noise
    x1 = trial.suggest_float("x1", -5, 10)
    x2 = trial.suggest_float("x2", 0, 15)
    noise_std = 0.1 * (1.0 + abs(x1) / 10 + abs(x2) / 15)
    return branin(x1, x2) + np.random.normal(0.0, noise_std)


def branin_noise_func(trial: optuna.trial.FrozenTrial) -> float:
    # Return the variance of the observation noise for a single-objective trial
    x1 = trial.params["x1"]
    x2 = trial.params["x2"]
    noise_std = 0.1 * (1.0 + abs(x1) / 10 + abs(x2) / 15)
    return noise_std**2


def multi_obj_noisy(trial: optuna.Trial) -> tuple[float, float]:
    # Multi-objective function with spatially-varying noise
    x = trial.suggest_float("x", 0, 1)
    y = trial.suggest_float("y", 0, 1)
    noise1_std = 0.05 * (1 + x)
    noise2_std = 0.05 * (1 + y)
    f1 = x**2 + np.random.normal(0, noise1_std)
    f2 = (x - 1) ** 2 + np.random.normal(0, noise2_std)
    return f1, f2


def multi_obj_noise_func(trial: optuna.trial.FrozenTrial) -> list[float]:
    # Return the per-objective variances as a list
    x = trial.params["x"]
    y = trial.params["y"]
    return [(0.05 * (1 + x)) ** 2, (0.05 * (1 + y)) ** 2]


if __name__ == "__main__":
    print("Running Single-Objective Heteroscedastic Optimization...")
    sampler = HeteroscedasticGPSampler(
        seed=42,
        noise_func=branin_noise_func,
        n_startup_trials=10,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(noisy_branin_objective, n_trials=50)

    print(f"Best value : {study.best_value:.4f}")
    print(f"Best params: {study.best_params}\n")

    print("Running Multi-Objective Heteroscedastic Optimization...")
    mo_sampler = HeteroscedasticGPSampler(
        seed=42,
        noise_func=multi_obj_noise_func,
        n_startup_trials=10,
    )
    mo_study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=mo_sampler,
    )
    mo_study.optimize(multi_obj_noisy, n_trials=50)

    pareto = mo_study.best_trials
    print(f"Number of Pareto-optimal trials: {len(pareto)}")
    print("Sample Pareto front (first 3):")
    for t in pareto[:3]:
        print(f"  params={t.params}, values={[round(v, 4) for v in t.values]}")
