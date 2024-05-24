---
author: 'Optuna team'
title: 'Random Search'
description: 'Sampler using random sampling.'
tags: ['sampler', 'built-in']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- RandomSampler

## Example
```python
import optuna
import optunahub


def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    return x**2


mod = optunahub.load_module("samplers/random_search")
sampler = mod.RandomSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10)
```

## Others
See the [documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html) for more details.

