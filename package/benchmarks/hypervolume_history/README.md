---
author: Shuhei Watanabe
title: Hypervolume History Calculator
description: This package computes the hypervolume history over completed trials in a multi-objective Optuna study, supporting any number of objectives, constraint handling, and both minimization and maximization directions.
tags: [benchmark, multi-objective, hypervolume]
optuna_versions: [4.8.0]
license: MIT License
---

## Abstract

This package provides a utility to compute the hypervolume history of a multi-objective Optuna study. The hypervolume indicator measures the volume of the objective space dominated by the current Pareto front relative to a reference point, and tracking it over trials is a common way to evaluate the convergence of multi-objective optimizers. The implementation supports arbitrary numbers of objectives, mixed minimization/maximization directions, and constraint handling via Optuna's constraint mechanism. For 2D and 3D cases, specialized efficient algorithms are used; for higher dimensions, the WFG algorithm is employed.

## APIs

### `get_hypervolume_history(study: optuna.Study, ref_point: np.ndarray | None = None) -> np.ndarray`

Compute the hypervolume history for a multi-objective study.

- `study`: An Optuna study with multiple objectives. Both `minimize` and `maximize` directions are supported.
- `ref_point`: The reference point for the hypervolume calculation. If `None`, a reference point is automatically inferred from the maximum observed values.

Returns a 1-D NumPy array of length equal to the number of completed trials, where the i-th element is the hypervolume of the Pareto front formed by the first i+1 completed trials. Infeasible trials (violating constraints) do not contribute to the hypervolume.

## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return (x + 2) ** 2 + (y + 2) ** 2, (x - 2) ** 2 + (y - 2) ** 2


sampler = optuna.samplers.TPESampler(seed=0)
study = optuna.create_study(sampler=sampler, directions=["minimize"] * 2)
study.optimize(objective, n_trials=200)

get_hypervolume_history = optunahub.load_module(
    "benchmarks/hypervolume_history"
).get_hypervolume_history
hv_history = get_hypervolume_history(study)
print(hv_history)
```
