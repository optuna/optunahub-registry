---
author: Shuhei Watanabe
title: A Sampler Using Parameter-Wise Bisection, aka Binary, Search
description: This sampler allows users to apply binary search to each parameter.
tags: [sampler, binary search, bisection search]
optuna_versions: [4.3.0]
license: MIT License
---

## Abstract

This sampler allows users to apply binary search to each parameter.
Please see the example for the usage.
Note that this sampler is not supposed to be used in the distributed optimization setup.

## APIs

- `BisectSampler(*, rtol: float = 1e-5, atol: float = 1e-8)`
  - `rtol`: The relative tolerance parameter to be used to judge whether all the parameters are converged. Default to that in `np.isclose`, i.e., `1e-5`.
  - `atol`: The absolute tolerance parameter to be used to judge whether all the parameters are converged. Default to that in `np.isclose`, i.e., `1e-8`.

By calling `BisectSampler.get_best_param(study)`, we can obtain the best parameter of the study.

## Installation

This sampler does not have any dependencies on top of `optunahub`.

## Example

For each parameter, please set `XXX_is_too_high` to the trial `user_attrs`.
Please see below for a concrete example.

```python
from collections.abc import Callable

import optuna
import optunahub


BisectSampler = optunahub.load_module("samplers/bisect").BisectSampler


def objective(trial: optuna.Trial, score_func: Callable[[optuna.Trial], float]) -> float:
    x = trial.suggest_float("x", -1, 1)
    # For each param, e.g., `ZZZ`, please set `ZZZ_is_too_high`.
    trial.set_user_attr("x_is_too_high", x > 0.5)
    y = trial.suggest_float("y", -1, 1, step=0.2)
    trial.set_user_attr("y_is_too_high", y > 0.2)
    # Please use `BisectSampler.score_func`.
    return BisectSampler.score_func(trial)


sampler = BisectSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=20)
```
