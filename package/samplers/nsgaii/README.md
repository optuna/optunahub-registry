---
author: Optuna team
title: NSGAII Search
description: Sampler using NSGAII algotithm.
tags: [sampler, built-in]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- NSGAIISampler

## Example

```python
import optuna
from optuna.samplers import NSGAIISampler


def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    return x**2


sampler = NSGAIISampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10)
```

## Others

See the [documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html) for more details.
