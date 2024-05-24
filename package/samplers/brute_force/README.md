---
author: 'Optuna team'
title: 'Brute Force Search'
description: 'Sampler using brute force.'
tags: ['sampler', 'built-in']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- BruteForceSampler

## Example
```python
import optuna
import optunahub


def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    return x**2


mod = optunahub.load_module("samplers/brute_force")
sampler = mod.BruteForceSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10)
```

## Others
See the [documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.BruteForceSampler.html) for more details.

