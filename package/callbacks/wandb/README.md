---
author: y0z, neel04, nabenabe0928, and nzw0301
title: Weights & Biases Callback
description: A callback to track Optuna trials with Weights & Biases.
tags: [callback, wandb, logging, built-in]
optuna_versions: [3.0.0]
license: MIT License
---

## Abstract

This callback enables tracking of Optuna study in Weights & Biases. The study is tracked as a single experiment run, where all suggested hyperparameters and optimized metrics are logged and plotted as a function of optimizer steps.

## APIs

- `WeightsAndBiasesCallback(metric_name: str | Sequence[str] = "value", wandb_kwargs: dict[str, Any] | None = None, as_multirun: bool = False)`
  - `metric_name`: Name assigned to optimized metric. In case of multi-objective optimization, list of names can be passed. Those names will be assigned to metrics in the order returned by objective function. If single name is provided, or this argument is left to default value, it will be broadcasted to each objective with a number suffix in order returned by objective function e.g. two objectives and default metric name will be logged as `value_0` and `value_1`. The number of metrics must be the same as the number of values objective function returns.
  - `wandb_kwargs`: Set of arguments passed when initializing Weights & Biases run. Please refer to [Weights & Biases API documentation](https://docs.wandb.ai/ref/python/init) for more details.
  - `as_multirun`: Creates new runs for each trial. Useful for generating W&B Sweeps like panels (for ex., parameter importance, parallel coordinates, etc).
- `WeightsAndBiasesCallback.track_in_wandb() -> Callable`
  - Decorator for using W&B for logging inside the objective function. The run is initialized with the same `wandb_kwargs` that are passed to the callback. All the metrics from inside the objective function will be logged into the same run which stores the parameters for a given trial. Use as `@wandbc.track_in_wandb()`.

## Installation

```shell
pip install wandb
```

## Example

### Add Weights & Biases callback to Optuna optimization.

```python
import optuna
import optunahub


module = optunahub.load_module("callbacks/wandb")
WeightsAndBiasesCallback = module.WeightsAndBiasesCallback


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


study = optuna.create_study()
wandbc = WeightsAndBiasesCallback(wandb_kwargs={"project": "my-project"})

study.optimize(objective, n_trials=10, callbacks=[wandbc])
```

### Weights & Biases logging in multirun mode

```python
import optuna
import optunahub
import wandb


module = optunahub.load_module("callbacks/wandb")
WeightsAndBiasesCallback = module.WeightsAndBiasesCallback

wandb_kwargs = {"project": "my-project"}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)


@wandbc.track_in_wandb()
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    wandb.log({"power": 2, "base of metric": x - 2}) # Additional logging in W&B.

    return (x - 2) ** 2


study = optuna.create_study()
study.optimize(objective, n_trials=10, callbacks=[wandbc])
```
