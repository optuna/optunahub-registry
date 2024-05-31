---
author: 'Optuna team'
title: 'Partial Fixed Sampler'
description: 'Sampler with partially fixed parameters.'
tags: ['sampler', 'built-in']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- PartialFixedSampler

## Example
```python
import optuna
from optuna.samplers import PartialFixedSampler


def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_int("y", -1, 1)
    return x**2 + y


tpe_sampler = optuna.samplers.TPESampler()
fixed_params = {"y": 0}
partial_sampler = PartialFixedSampler(fixed_params, tpe_sampler)

study = optuna.create_study(sampler=partial_sampler)
study.optimize(objective, n_trials=10)
```

## Others
See the [documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.PartialFixedSampler.html) for more details.
