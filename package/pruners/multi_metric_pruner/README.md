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

Two reporting modes are available (do **not** mix them within the same trial):

| Mode | Report call | `prune` / `should_prune` |
|---|---|---|
| Multi-metric | `trial.report([v1, v2, ...], step)` | `metric_name=None` (default) |
| Single-metric | `trial.report(v, step, metric_name="loss")` | `metric_name="loss"` |

### Multi-metric mode

All metrics are reported together at each step. The pruner ranks every trial at each step
using Pareto dominance (following Optuna's MOTPE tie-breaking via hypervolume subset
selection). The resulting Pareto ranks serve as single-metric intermediate values passed
to the base pruner.

### Single-metric mode

Each metric is reported independently by name. The pruner extracts the values for the
specified metric and passes them verbatim to the base pruner. Different metrics can be
pruned at different steps using different calls to `should_prune`.

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
        trial.report([metric1, metric2], step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return x**2, (x - 2.0) ** 2


study = optuna.create_study(
    directions=["minimize", "minimize"],
    pruner=MultiMetricPruner(
        optuna.pruners.MedianPruner(n_startup_trials=3),
        directions=["minimize", "minimize"],
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
| `directions` | `list[str] \| None` | `None` | Directions (`"minimize"` / `"maximize"`) for each metric in multi-metric mode. Defaults to all-minimize when `None`. |
| `metric_directions` | `dict[str, str] \| None` | `None` | Per-metric directions for single-metric mode, e.g. `{"loss": "minimize", "accuracy": "maximize"}`. Unlisted metrics default to `"minimize"`. |

### `prune(study, trial, *, metric_name=None)`

The `prune` method extends `BasePruner.prune` with an optional `metric_name` keyword
argument. Pass `metric_name` to select a specific metric when using single-metric mode.
