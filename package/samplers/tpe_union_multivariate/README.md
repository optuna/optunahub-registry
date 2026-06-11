---
author: Rishabh Dewangan
title: TPE Union Multivariate Sampler
description: A multivariate TPE sampler utilizing a union search space approach to support conditional distributions cleanly without independent group decomposition.
tags: [sampler]
optuna_versions: [4.7.0]
license: MIT License
---

## Abstract

This package implements a union search space approach for the multivariate TPE sampler. It enables optimization across conditional or dynamic configurations without forcing independent fallback routines or dropping back to intersection group decompositions. By padding unobserved states with `np.nan` and dynamically managing categorical distribution dimensions, the sampler successfully estimates a single joint distribution that models correlations between top-level routing choices and nested hyperparameters.

## APIs

- `TPEUnionMultivariateSampler(*, n_startup_trials: int = 10, seed: int | None = None)`
  - `n_startup_trials`: The minimum number of completed trials required before the sampler switches from random initialization to Parzen GMM estimation.
  - `seed`: Seed configuration parameter for the internal random number generator to ensure reproducible execution trajectories.

## Example

```python
import optuna
import optunahub

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_categorical("x", [True, False])
    y = trial.suggest_float("y", -1, 1)
    if x:
        a = trial.suggest_float("a", -1, 1)
        return (a - y) ** 2
    else:
        b = trial.suggest_float("b", -1, 1)
        return (b - y) ** 2

if __name__ == "__main__":
    module = optunahub.load_module(package="samplers/tpe_union_multivariate")
    sampler = module.TPEUnionMultivariateSampler(n_startup_trials=5, seed=42)
    
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=30)
    print(f"Best trial value: {study.best_value}")
```

## Others

### Empirical Convergence Validation

The following benchmark demonstrates the optimization tracking capabilities evaluated against a hierarchical categorical landscape running 32 repetitions across 150 trials:

![Conditional Convergence Chart](https://github.com/user-attachments/assets/be4a5b2f-a213-4842-8beb-8c504fa21bdd)

The tracking results confirm that estimating unified conditional search spaces avoids local optimization plateaus and stabilizes overall convergence trajectories on nested parameters.
