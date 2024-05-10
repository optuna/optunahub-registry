---
author: 'Optuna team'
title: 'Demo Sampler'
description: 'Demo Sampler of OptunaHub'
tags: ['sampler']
optuna_versions: [3.6.1]
license: 'MIT'
lastmod: 2024-04-22T18:42:26+09:00
---

This package provides a demo sampler of OptunaHub.

## class DemoSampler(seed)

### Parameters
- `seed: int` - A random seed


## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 0, 1)

    return x


if __name__ == "__main__":
    module = optunahub.load("samplers/demo")
    sampler = module.DemoSampler(seed=42)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=5)

    print(study.best_trial)
```
