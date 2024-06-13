---
author: 'Optuna team'
title: 'Percentile Pruner'
description: 'Pruner to keep the specified percentile of the trials.'
tags: ['pruner', 'built-in']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- PercentilePruner

## Example
```python
import optuna
from optuna.pruners import PercentilePruner


def objective(trial):
    s = 0
    for step in range(20):
       x = trial.suggest_float(f"x_{step}", -5, 5)
       s += x**2
       trial.report(s, step)
       if trial.should_prune():
            raise optuna.TrialPruned()
    return s


pruner = PercentilePruner(25.0)
study = optuna.create_study(pruner=pruner)
study.optimize(objective, n_trials=20)
```

## Others
See the [documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PercentilePruner.html) for more details.

