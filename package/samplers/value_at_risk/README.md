---
author: Shuhei Watanabe
title: Robust Bayesian Optimization under Input Noise (VaR, Value at Risk)
description: This sampler searches for parameters robust to input noise
tags: [sampler, gp, bo]
optuna_versions: [4.5.0]
license: MIT License
---

## Abstract

This sampler optimizes the objective function given input perturbations.
For example, most industrial productions have their own tolerance, and any differences within this tolerance are considered acceptable.
For this reason, even if we transfer the optimized result of a design simulation to its production, it is mostly impossible to reproduce the precise simulated design setup.
This necessitates accounting for noisy input so that the deployed setup remains sufficiently performant even in the presence of arbitrary noise.
To implement this sampler, the author referred to [this paper](https://arxiv.org/abs/2202.07549).

## APIs

- `RobustGPSampler(*, seed: int | None = None, independent_sampler: BaseSampler | None = None, n_startup_trials: int = 10, deterministic_objective: bool = False, constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None, warn_independent_sampling: bool = True, uniform_input_noise_ranges: dict[str, float] | None = None, normal_input_noise_stdevs: dict[str, float] | None = None)`
  - `seed`: Random seed to initialize internal random number generator. Defaults to `None` (a seed is picked randomly).
  - `independent_sampler`: Sampler used for initial sampling (for the first `n_startup_trials` trials) and for conditional parameters. Defaults to :obj:`None` (a random sampler with the same `seed` is used).
  - `n_startup_trials`: Number of initial trials. Defaults to 10.
  - `deterministic_objective`: Whether the objective function is deterministic or not. If `True`, the sampler will fix the noise variance of the surrogate model to the minimum value (slightly above 0 to ensure numerical stability). Defaults to `False`. Currently, all the objectives will be assume to be deterministic if `True`.
  - `constraints_func`: An optional function that computes the objective constraints. It must take a `optuna.trial.FrozenTrial` and return the constraints. The return value must be a sequence of `float`. A value strictly larger than 0 means that a constraint is violated. A value equal to or smaller than 0 is considered feasible. If `constraints_func` returns more than one value for a trial, that trial is considered feasible if and only if all values are equal to 0 or smaller. The `constraints_func` will be evaluated after each successful trial. The function won't be called when trials fail or are pruned, but this behavior is subject to change in future releases.
  - `warn_independent_sampling`: If this is `True`, a warning message is emitted when the value of a parameter is sampled by using an independent sampler, meaning that no GP model is used in the sampling. Note that the parameters of the first trial in a study are always sampled via an independent sampler, so no warning messages are emitted in this case.
  - `uniform_input_noise_ranges`: The input noise ranges for each parameter. For example, when `{"x": 0.1, "y": 0.2}`, the sampler assumes that $\\pm$ 0.1 is acceptable for `x` and $\\pm$ 0.2 is acceptable for `y`.
  - `normal_input_noise_stdevs`: The input noise standard deviations for each parameter. For example, when `{"x": 0.1, "y": 0.2}` is given, the sampler assumes that the input noise of `x` and `y` follows `N(0, 0.1**2)` and `N(0, 0.2**2)`, respectively.

Please note that only one of `uniform_input_noise_ranges` and `normal_input_noise_stdevs` can be provided.

## Installation

The dependencies will be installed either via:

```shell
$ pip install scipy
$ pip install torch --index-url https://download.pytorch.org/whl/cpu
```

or

```shell
$ pip install -r https://hub.optuna.org/samplers/value_at_risk/requirements.txt
```

## Example

```python
from __future__ import annotations

import numpy as np
import optuna
import optunahub


def numpy_objective(X: np.ndarray) -> np.ndarray:
    C = np.asarray([[2.0, -12.2, 21.2, -6.4, -4.7, 6.2], [1.0, -11, 43.3, -74.8, 56.9, -10]])
    X_poly = np.zeros_like(X)
    for i in range(C.shape[1]):
        X_poly = X * (X_poly + C[:, i])
    X0X1 = X[..., 0] * X[..., 1]
    out = np.sum(X_poly, axis=-1)
    out += X0X1 * (-4.1 - 0.1 * X0X1 + 0.4 * X[..., 0] + 0.4 * X[..., 1])
    return out


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -1, 4)
    y = trial.suggest_float("y", -1, 4)
    X = np.array([x, y])
    f = numpy_objective(X).item()
    return f


RobustGPSampler = optunahub.load_local_module(
    "samplers/value_at_risk", registry_root="./package"
).RobustGPSampler
sampler = RobustGPSampler(seed=0, uniform_input_noise_ranges={"x": 0.5, "y": 0.5})
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=50)

```
