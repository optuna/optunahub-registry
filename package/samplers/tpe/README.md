---
author: Optuna team
title: TPE Sampler
description: Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
tags: [sampler, built-in]
optuna_versions: [5.1.0.dev]
license: MIT License
---

## Class or Function Names

- TPESampler

## Example

```python
import optuna
from optuna.samplers import TPESampler


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return x**2


sampler = TPESampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10)
```

## Categorical distance

This package is based on the Optuna 5.1.0.dev `TPESampler` and restores categorical-distance
support using the Optuna 4.9 implementation as a behavioral reference. 

## Others

See the [documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) for more details.
