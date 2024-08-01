---
author: mist714
title: Sampler using Whale Optimization Algorithm
description: Swarm Algorithm Inspired by Pod of Whale
tags: [sampler]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- WhaleOptimizationSampler

## Example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/optuna/optunahub-registry/blob/main/package/samplers/whale_optimization/example.ipynb)

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import optuna
import optunahub


WhaleOptimizationSampler = optunahub.load_module(
    "samplers/whale_optimization"
).WhaleOptimizationSampler

if __name__ == "__main__":

    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x**2 + y**2

    sampler = WhaleOptimizationSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.show()
```

## Others

### Reference

Mirjalili, Seyedali & Lewis, Andrew. (2016). The Whale Optimization Algorithm. Advances in Engineering Software. 95. 51-67. 10.1016/j.advengsoft.2016.01.008.
