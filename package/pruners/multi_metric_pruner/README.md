---
author: Shuhei Watanabe
title: Multi-Metric Pruner
description: Pruner that supports intermediate value reporting for multi-objective optimization, using Pareto ranking (multi-metric mode) or named per-metric pruning (single-metric mode).
tags: [pruner, multi-objective, intermediate, pruning]
optuna_versions: [4.8.0]
license: MIT License
---

## Abstract

Optuna's built-in `trial.report()` raises `NotImplementedError` in multi-objective studies.
`MultiMetricPruner` works around this by storing intermediate values in trial user attributes
and constructing a synthetic single-objective study for the wrapped base pruner to evaluate.

The pruning mode is selected via the `joint` argument:

| Mode         | `joint` | `report` call (Example with `metric_names = ["loss", "acc"]`)                      |
| ------------ | ------- | ---------------------------------------------------------------------------------- |
| Multi-metric | `True`  | `trial.report({"loss": v1, "acc": v2}, step)`                                      |
| Per-metric   | `False` | `trial.report({"loss": v1, "acc": v2}, step)` or `trial.report({"loss": v}, step)` |

### Multi-metric mode (`joint=True`)

All metrics are reported together as a dict at each step. The pruner ranks every trial at
each step using Pareto dominance. The resulting Pareto ranks serve as single-metric
intermediate values passed to the base pruner.

### Per-metric mode (`joint=False`)

Each metric is evaluated independently by the base pruner. Calling `should_prune()` with no
argument checks all metrics and prunes if any one of them triggers the base pruner. You can
also pass `metric_name` to `should_prune()` to restrict the check to a single metric.
This mode supports mixed-frequency reporting where different metrics are reported at
different step intervals.

This is convenient when each objective has different computational overhead or when we would like to track multiple metrics per objective.

For example, we often encounter the following example in LLM trainings:

```python
def objective(trial: optuna.Trial) -> tuple[float, float]:
    mmt = MultiMetricPrunerTrial(trial)
    lr = mmt.suggest_float("lr", 1e-6, 1e-4, log=True)
    train_data_loader = ...
    val_data_loader = ...
    best_val_loss = ...
    for epoch in range(10):
        for step, batch in data_loader:
            train_loss = ...
            mmt.report({"train_loss": train_loss}, step=step)
            if mmt.should_prune(metric_name="train_loss"):
                raise optuna.TrialPruned()
        val_loss = ...
        for i, batch in val_loader:
            ...
        mmt.report({"val_loss": val_loss}, step=epoch)
        if mmt.should_prune(metric_name="val_loss"):
            raise optuna.TrialPruned()
        best_val_loss = min(val_loss, best_val_loss)
    return best_val_loss
```

## APIs

- `MultiMetricPruner(base_pruner, *, metric_directions, joint)`
  - `base_pruner`: Pruner that makes the actual pruning decision.
  - `metric_directions`: Mapping from metric name to direction (`"minimize"` / `"maximize"`).
  - `joint`: If `True`, use multi-metric (Pareto-rank) mode. If `False`, use per-metric mode where each metric is evaluated independently.
- `MultiMetricPrunerTrial(trial)`
  - `trial`: The trial object received in the objective function.

## Example

```python
import optuna
import optunahub

module = optunahub.load_module("pruners/multi_metric_pruner")
MultiMetricPruner = module.MultiMetricPruner
MultiMetricPrunerTrial = module.MultiMetricPrunerTrial


def objective(trial: optuna.Trial) -> tuple[float, float]:
    mmt = MultiMetricPrunerTrial(trial)
    x = mmt.suggest_float("x", -5.0, 5.0)
    for step in range(10):
        metric1 = (x - step * 0.1) ** 2
        metric2 = (x + step * 0.1) ** 2
        mmt.report({"loss": metric1, "acc": metric2}, step)
        if mmt.should_prune():
            raise optuna.TrialPruned()
    return x**2, (x - 2.0) ** 2


study = optuna.create_study(
    directions=["minimize", "minimize"],
    pruner=MultiMetricPruner(
        optuna.pruners.MedianPruner(n_startup_trials=3),
        metric_directions={"loss": "minimize", "acc": "minimize"},
        joint=True,
    ),
)
study.optimize(objective, n_trials=30)
```

See [example.py](https://github.com/optuna/optunahub-registry/blob/main/package/pruners/multi_metric_pruner/example.py) for a full example including per-metric and mixed-frequency modes.
