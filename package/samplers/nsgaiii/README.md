---
author: 'Optuna team'
title: 'NSGAIII Search'
description: 'Sampler using NSGAIII algotithm.'
tags: ['sampler', 'built-in']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- NSGAIIISampler

## Example
```python
import optuna
import optunahub


def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    return x**2


mod = optunahub.load_module("samplers/nsgaiii")
sampler = mod.NSGAIIISampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10)
```

## Others
See the [documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIIISampler.html) for more details.

