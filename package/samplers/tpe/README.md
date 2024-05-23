---
author: 'Optuna team'
title: 'TPE Sampler'
description: 'Sampler using TPE (Tree-structured Parzen Estimator) algorithm.'
tags: ['sampler', 'built-in']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- TPESampler

## Example
```python
import optuna
import optunahub


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return x**2


mod = optunahub.load_module("samplers/tpe")
sampler = mod.TPESampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10)
```

## Others
See the [documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) for more details.
