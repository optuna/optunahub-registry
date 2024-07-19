---
author: Optuna team
title: Threshold Pruner
description: Pruner to detect outlying metrics of the trials.
tags: [pruner, built-in]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- ThresholdPruner

## Example

```python
from optuna import create_study
from optuna.pruners import ThresholdPruner
from optuna import TrialPruned


def objective_for_upper(trial):
    for step, y in enumerate(ys_for_upper):
        trial.report(y, step)

        if trial.should_prune():
            raise TrialPruned()
    return ys_for_upper[-1]


def objective_for_lower(trial):
    for step, y in enumerate(ys_for_lower):
        trial.report(y, step)

        if trial.should_prune():
            raise TrialPruned()
    return ys_for_lower[-1]


ys_for_upper = [0.0, 0.1, 0.2, 0.5, 1.2]
ys_for_lower = [100.0, 90.0, 0.1, 0.0, -1]

study = create_study(pruner=ThresholdPruner(upper=1.0))
study.optimize(objective_for_upper, n_trials=10)

study = create_study(pruner=ThresholdPruner(lower=0.0))
study.optimize(objective_for_lower, n_trials=10)
```

## Others

See the [documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.ThresholdPruner.html) for more details.
