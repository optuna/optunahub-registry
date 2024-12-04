---
author: Shuhei Watanabe
title: NSGAIISampler Using TPESampler for the Initialization
description: This sampler uses TPESampler for the initialization to warm start.
tags: [sampler, tpe, nsgaii, warmstart]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

This sampler uses `TPESampler` instead of `RandomSampler` for the initialization of `NSGAIISampler`.

## APIs

- NSGAIIWithTPEWarmupSampler

This class takes the identical interface as the Optuna [`NSGAIISampler`](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html).

## Example

```python
from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2, (x - 2) ** 2 + (y - 2) ** 2


package_name = "samplers/nsgaii_with_tpe_warmup"
sampler = optunahub.load_module(package=package_name).NSGAIIWithTPEWarmupSampler()
study = optuna.create_study(sampler=sampler, directions=["minimize"]*2)
study.optimize(objective, n_trials=60)

```
