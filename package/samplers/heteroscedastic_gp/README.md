---
author: Rishabh-git10
title: Heteroscedastic Gaussian-Process Sampler
description: A Gaussian Process-based Bayesian Optimization sampler that natively supports heteroscedastic (input-dependent) observation noise to prevent surrogate model corruption.
tags: [sampler, gaussian-process, bayesian-optimization, heteroscedastic]
optuna_versions: [3.6.0]
license: MIT License
---

## Abstract

Standard Bayesian Optimization assumes homoscedastic (constant) noise across the entire search space. When standard Gaussian Process (GP) models encounter a highly noisy localized region, the Marginal Log-Likelihood optimization is forced to absorb that localized variance into a single global noise parameter. This inflates uncertainty across the entire surrogate model, causing the optimizer to over-explore and waste search budget.

The `HeteroscedasticGPSampler` natively supports input-dependent observation noise. By explicitly passing the known or estimated noise variance of a trial via a user-defined `noise_func`, this sampler maps and isolates high-variance regions. This prevents global surrogate corruption, allowing the acquisition function to maintain sharp exploitation in the clean regions of the search space.

This mirrors the architectural capability of advanced frameworks like BoTorch (via `train_Yvar`), bringing robust, noise-aware Bayesian Optimization natively to Optuna.

## Performance

When evaluated on the Branin-Hoo function injected with input-dependent variance, the `HeteroscedasticGPSampler` significantly reduces median regret compared to the standard `GPSampler`. Explicitly mapping the noise variance allows the surrogate model to avoid over-exploring locally noisy regions, accelerating convergence on the true optimum.

## APIs

- `HeteroscedasticGPSampler(*, seed: int | None = None, independent_sampler: BaseSampler | None = None, n_startup_trials: int = 10, deterministic_objective: bool = False, constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None, noise_func: Callable[[FrozenTrial], float | Sequence[float]] | None = None, warn_independent_sampling: bool = True)`
  - `noise_func`: An optional function that computes the observation noise variance for a trial. It must take a `optuna.trial.FrozenTrial` and return a `float` (for single-objective optimization) or a sequence of `float`s (for multi-objective optimization). The returned value must be the **variance** of the noise, not the standard deviation. If not provided, the sampler behaves identically to the standard `GPSampler`.
  - `constraints_func`: An optional function that computes the objective constraints, returning a sequence of floats where values `<= 0` are considered feasible.
  - `deterministic_objective`: Whether the objective function is deterministic. If `True`, the sampler fixes the noise variance of the surrogate model to the minimum value.
  - `n_startup_trials`: Number of initial trials evaluated using the `independent_sampler` (default is RandomSearch) before the GP model starts generating suggestions.

## Installation

This package relies on the core dependencies of Optuna's native GP module. OptunaHub will automatically resolve the `requirements.txt` when loaded, or you can install them manually:

```shell
pip install scipy torch
```

## Example

To use this sampler, define a `noise_func` that accepts a completed `FrozenTrial` and returns the variance of the observation noise.

```python
import optuna
import optunahub
import numpy as np


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 5.0)
    
    # Noise variance is a known function of the input
    noise_variance = 0.1 * np.exp(abs(x))
    
    # Log the variance so it can be retrieved by the sampler
    trial.set_user_attr("variance", noise_variance)
    
    true_mean = x**2
    return true_mean + np.random.normal(0, np.sqrt(noise_variance))


def my_noise_func(trial: optuna.trial.FrozenTrial) -> float:
    # Retrieve the variance logged during the objective evaluation
    return trial.user_attrs.get("variance", 1e-5)


if __name__ == "__main__":
    module = optunahub.load_module("samplers/heteroscedastic_gp")
    HeteroscedasticGPSampler = module.HeteroscedasticGPSampler

    study = optuna.create_study(
        direction="minimize",
        sampler=HeteroscedasticGPSampler(
            noise_func=my_noise_func,
            n_startup_trials=10,
        )
    )
    study.optimize(objective, n_trials=50)

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")
```

## Others

### Supported Features

- Single-Objective Optimization: (LogEI)

- Multi-Objective Optimization: (LogEHVI) - Return a list of variances from `noise_func`.

- Constrained Optimization: Fully compatible with `constraints_func`.

- Deterministic Optimization: Set `deterministic_objective=True`.

## Reference

- Makarova, A., Aschenbrenner, M., Fiducioso, M., Krause, A. (2021). Risk-averse Heteroscedastic Bayesian Optimization. Advances in Neural Information Processing Systems (NeurIPS).
- Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. Chapter 9: Incorporating Explicit Noise Models.
- Architecture Discussion and Context: [optuna/optuna#6457](https://github.com/optuna/optuna/issues/6457)
