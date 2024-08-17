---
author: k-onoue
title: Sampler based on Grey Wolf Optimization (GWO)
description: Metaheuristic algorithm inspired by the leadership and hunting behavior of grey wolves
tags: [sampler]
optuna_versions: [3.6.1]
license: "MIT License"
---

## Class or Function Names

- GreyWolfOptimizationSampler

## Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/optuna/optunahub-registry/blob/main/package/samplers/grey_wolf_optimization/example.ipynb)

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import optuna
import optunahub


GreyWolfOptimizationSampler = optunahub.load_module(
    "samplers/grey_wolf_optimization"
).GreyWolfOptimizationSampler

if __name__ == "__main__":

    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y**2

    sampler = GreyWolfOptimizationSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.show()
```

## Others

### Reference

Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. Advances in engineering software, 69, 46-61.