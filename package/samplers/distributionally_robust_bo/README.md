---
author: Rishabh Dewangan
title: Distributionally Robust BO Sampler
description: A sampler based on Distributionally Robust Bayesian Optimization for chance-constrained problems.
tags: [sampler]
optuna_versions: [3.6.1]
license: MIT License
---

## Abstract

This package provides a sampler based on the algorithm proposed in **"Bayesian Optimization for Distributionally Robust Chance-constrained Problem"** (ICML 2022).

Standard Bayesian Optimization assumes a known distribution of environmental variables. However, in many real-world scenarios (like manufacturing tolerances or financial markets), the true distribution is uncertain. This sampler uses a **Distributionally Robust Chance Constraint (DRCC)** to ensure that the objective function satisfies a specific threshold ($g(x, w) > h$) with at least a probability $\alpha$, even under the worst-case probability distribution within a defined ambiguity set.

## APIs

- `DistributionallyRobustSampler(*, epsilon_t: float = 0.15, h: float = 5.0, alpha: float = 0.53, seed: int | None = None)`
  - `epsilon_t`: The radius of the ambiguity set. It defines how much the worst-case probability distribution can deviate from the reference distribution. A higher value means a more robust (but more conservative) optimization.
  - `h`: The threshold value for the chance constraint.
  - `alpha`: The minimum required probability that the function value will exceed the threshold `h`.
  - `seed`: Seed for the random number generator, used primarily during the initial exploration phase.

## Example

```python
import optuna
import optunahub

def objective(trial: optuna.Trial) -> float:
    # A simple 2D objective function
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    return x**2 + y**2

# Load the sampler from OptunaHub
sampler = optunahub.load_module(
    package="samplers/distributionally_robust_bo"
).DistributionallyRobustSampler(
    epsilon_t=0.15, 
    h=5.0, 
    alpha=0.53
)

study = optuna.create_study(sampler=sampler, direction="minimize")
study.optimize(objective, n_trials=30)

print(study.best_trials)
```

## Reference

Hideaki Imamura, et al. "Bayesian Optimization for Distributionally Robust Chance-constrained Problem." Proceedings of the 39th International Conference on Machine Learning (ICML). 2022.
