---
author: Shuhei Watanabe
title: CMA-ES with User Prior
description: You can provide the initial parameters, i.e. mean vector and covariance matrix, for CMA-ES with this sampler.
tags: [sampler, cma-es, meta-learning, prior]
optuna_versions: [4.0.0]
license: MIT License
---

## Abstract

As the Optuna CMA-ES sampler does not support any flexible ways to initialize the parameters of the Gaussian distribution, so I created a workaround to do so.

## Class or Function Names

- UserPriorCmaEsSampler

## Installation

```shell
$ pip install optunahub cmaes
```

## Example

The simplest code example is as follows:

```python
import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -50, -40)
    y = trial.suggest_int("y", -5, 5)
    return (x + 43)**2 + (y - 2)**2


if __name__ == "__main__":
    module = optunahub.load_module(package="samplers/user_prior_cmaes")
    sampler = module.UserPriorCmaEsSampler(param_names=["x", "y"], mu0=np.array([3., -48.]), cov0=np.diag([0.2, 2.0]))
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)
    print(study.best_trial.value, study.best_trial.params)

```

Although `UserPriorCmaEsSampler` CANNOT support log scale from the sampler side, we have a workaround to do so:

```python
import math

import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    # For example, trial.suggest_float("x", 1e-5, 1.0, log=True) can be encoded as:
    x = 10 ** trial.suggest_float("log10_x", -5, 0)
    # trial.suggest_float("y", 2, 1024, log=True) can be encoded as:
    y = 2 ** trial.suggest_float("log2_y", 1, 10)
    # In general, trial.suggest_float("z", low, high, log=True) can be encoded as:
    low, high = 3, 81
    b = 3  # The base of log can be any positive number.
    z = b ** trial.suggest_float("logb_z", math.log(low, b), math.log(high, b))
    return x**2 + y**2 + z**2


if __name__ == "__main__":
    module = optunahub.load_module(package="samplers/user_prior_cmaes")
    sampler = module.UserPriorCmaEsSampler(
        param_names=["log10_x", "log2_y", "logb_z"],
        mu0=np.array([-4, 8, 3]),
        cov0=np.diag([0.2, 1., 0.1]),
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)
    print(study.best_trial.value, study.best_trial.params)
```
