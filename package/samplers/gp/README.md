---
author: 'Optuna team'
title: 'Gaussian Process-Based Sampler'
description: 'Sampler using Gaussian process-based Bayesian optimization.'
tags: ['sampler', 'built-in']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- GPSampler

## Installation
```bash
pip install scipy pytorch
```

## Example
```python
import optuna
import optunahub


def objective(trial):
  x = trial.suggest_float("x", -5, 5)
  return x**2


mod = optunahub.load_module("samplers/gp")
sampler = mod.GPSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)
```

## Others
See the [documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GPSampler.html) for more details.
