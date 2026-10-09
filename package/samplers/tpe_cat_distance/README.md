---
author: Optuna team
title: TPE Sampler
description: Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
tags: [sampler, built-in]
optuna_versions: [5.1.0.dev]
license: MIT License
---

## Class or Function Names

- TPESampler

## APIs

- `TPESampler(...)`
  - This is Optuna's TPE sampler with support for
    `categorical_distance_func`.
  - `categorical_distance_func`: A mapping from categorical parameter names to distance
    functions. Each function receives two choices and returns a non-negative distance.

## Example

```python
import optuna
import optunahub


module = optunahub.load_module(package="samplers/tpe_cat_distance")


def hamming_distance(x: str, y: str) -> float:
    return float(x != y)


sampler = module.TPESampler(
    categorical_distance_func={"optimizer": hamming_distance},
)


study = optuna.create_study(sampler=sampler)


def objective(trial):
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    return float(optimizer != "adam")


study.optimize(objective, n_trials=20)
```

## Categorical distance

This package is based on the Optuna 5.1.0.dev `TPESampler` and restores categorical-distance
support using the Optuna 4.9 implementation as a behavioral reference.

## Others

See the [documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) for more details.
