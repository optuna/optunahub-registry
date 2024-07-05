---
author: Optuna team
title: CMA-ES Sampler
description: A sampler using cmaes as the backend.
tags: [sampler, built-in]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- CmaEsSampler

## Installation

```bash
pip install cmaes
```

## Example

```python
import optuna
from optuna.samplers import CmaEsSampler


def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_int("y", -1, 1)
    return x**2 + y


sampler = CmaEsSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=20)
```

## Others

See the [documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html) for more details.
