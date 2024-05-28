---
author: 'Optuna team'
title: 'Grid Search'
description: 'Sampler using grid search.'
tags: ['sampler', 'built-in']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- GridSampler

## Example
```python
import optuna
from optuna.samplers import GridSampler


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_int("y", -100, 100)
    return x**2 + y**2


search_space = {"x": [-50, 0, 50], "y": [-99, 0, 99]}
sampler = GridSampler(search_space)
study = optuna.create_study(sampler=sampler)
study.optimize(objective)
```

## Others
See the [documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GridSampler.html) for more details.
