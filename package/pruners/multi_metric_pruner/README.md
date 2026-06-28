---
author: Shuhei Watanabe
title: Multi-Metric Pruner
description: Pruner that supports intermediate value reporting for multi-objective optimization, using Pareto ranking (multi-metric mode) or named per-metric pruning (single-metric mode).
tags: [pruner, multi-objective, intermediate]
optuna_versions: [4.8.0]
license: MIT License
---

## Class or Function Names

- `MultiMetricPruner`
- `MultiMetricPrunerTrial`

## Overview

Optuna's built-in `trial.report()` raises `NotImplementedError` in multi-objective studies.
`MultiMetricPruner` works around this by storing intermediate values in trial user attributes
and constructing a synthetic single-objective study for the wrapped base pruner to evaluate.

The pruning mode is determined by the number of entries in `metric_directions`:

| Mode | `metric_directions` length | `report` call |
|---|---|---|
| Multi-metric | > 1 | `trial.report({"loss": v1, "acc": v2}, step)` |
| Single-metric | 1 | `trial.report({"loss": v}, step)` |

### Multi-metric mode

All metrics are reported together as a dict at each step. The pruner ranks every trial at
each step using Pareto dominance. The resulting Pareto ranks serve as single-metric
intermediate values passed to the base pruner.

### Single-metric mode

A single metric is reported by name. The pruner extracts that metric's values and passes
them verbatim to the base pruner.

## Example

```python
import optuna
import optunahub

module = optunahub.load_local_module("pruners/multi_metric_pruner", registry_root="package/")
MultiMetricPruner = module.MultiMetricPruner
MultiMetricPrunerTrial = module.MultiMetricPrunerTrial


def objective(trial: optuna.Trial) -> tuple[float, float]:
    trial = MultiMetricPrunerTrial(trial)
    x = trial.suggest_float("x", -5.0, 5.0)
    for step in range(10):
        metric1 = (x - step * 0.1) ** 2
        metric2 = (x + step * 0.1) ** 2
        trial.report({"loss": metric1, "acc": metric2}, step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return x**2, (x - 2.0) ** 2


study = optuna.create_study(
    directions=["minimize", "minimize"],
    pruner=MultiMetricPruner(
        optuna.pruners.MedianPruner(n_startup_trials=3),
        metric_directions={"loss": "minimize", "acc": "minimize"},
    ),
)
study.optimize(objective, n_trials=30)
```

See [example.py](https://github.com/optuna/optunahub-registry/blob/main/package/pruners/multi_metric_pruner/example.py) for a full example including single-metric mode.

## Arguments

### `MultiMetricPruner`

| Argument | Type | Default | Description |
|---|---|---|---|
| `base_pruner` | `BasePruner` | — | Pruner that makes the actual pruning decision. |
| `metric_directions` | `dict[str, str]` | — | Mapping from metric name to direction (`"minimize"` / `"maximize"`). Pass multiple entries for multi-metric (Pareto-rank) mode, or a single entry for single-metric mode. |
