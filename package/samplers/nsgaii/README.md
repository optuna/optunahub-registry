---
author: 'Optuna team'
title: 'NSGAII Search'
description: 'Sampler using NSGAII algotithm.'
tags: ['sampler', 'built-in']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- NSGAIISampler

## Example
```python
import optuna
import optunahub


def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    return x**2


mod = optunahub.load_module("samplers/nsgaii")
sampler = mod.NSGAIISampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10)
```

## Others
See the [documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html) for more details.

