---
author: Shuhei Watanabe
title: Multivariate TPE Sampler that considers dynamic value ranges
description: Multivariate TPESampler that includes past observations even after search space changes.
tags: [sampler, tpe, dynamic]
optuna_versions: [4.9.0]
license: MIT License
---

## Abstract

...

## Class or Function Names

- TPESampler

## Installation

```shell
$ pip install -r https://hub.optuna.org/samplers/multivariate_tpe_flex/requirements.txt
```

## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


module = optunahub.load_module(package="samplers/multivariate_tpe_flex")
sampler = module.TPESampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)
```
