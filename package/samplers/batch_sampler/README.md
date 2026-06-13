---
author: Mark Shipman
title: Batch Sampler
description: Coordinates parallel Optuna workers into batches so each worker receives a distinct, jointly-selected suggestion rather than an independent one.
tags: [sampler, batch, parallel]
optuna_versions: [4.8.0]
license: MIT License
---

## Abstract

`BatchSampler` solves a coordination problem that arises when running Optuna
with multiple workers (`n_jobs > 1`).

With default samplers, each worker calls `sample_relative` concurrently and
sees the same incomplete view of the trial history — pending trials from the
same batch are invisible to one another.  This causes workers to suggest
near-duplicate configurations, wasting the throughput gained from parallelism.

`BatchSampler` uses a shared lock to coordinate workers: the first worker to
find an empty cache calls a user-supplied `suggest_fn` once to obtain `q`
suggestions jointly, then hands them out one at a time.  Other workers wait
on the lock and pop from the already-filled cache without triggering another
call.  The result is a batch of `q` diverse, coordinated suggestions rather
than `q` independent ones.

`suggest_fn` can be any callable that takes the current observations and
returns `q` parameter dicts — a local Bayesian optimisation function, a GP
fitted in-process, or a wrapper around an external service.  The sampler
imposes no constraints on how suggestions are generated.

During startup (fewer than `n_startup_trials` complete trials) the sampler
falls back to random search automatically.

## APIs

### `DimSpec(name, type, low, high, log=False, step=None)`

Dataclass describing one dimension of the search space.

| Field  | Type             | Description                                               |
| ------ | ---------------- | --------------------------------------------------------- |
| `name` | `str`            | Parameter name (must match `trial.suggest_*` calls).      |
| `type` | `"float"\|"int"` | Distribution family.                                      |
| `low`  | `float`          | Lower bound (inclusive).                                  |
| `high` | `float`          | Upper bound (inclusive).                                  |
| `log`  | `bool`           | Use log-uniform spacing. Default `False`.                 |
| `step` | `float\|None`    | Grid step for `int` dims (default 1). Unused for `float`. |

### `BatchSampler(search_space, suggest_fn, ...)`

| Argument           | Type            | Default | Description                                                                   |
| ------------------ | --------------- | ------- | ----------------------------------------------------------------------------- |
| `search_space`     | `list[DimSpec]` | —       | **Required.** Dimensions of the optimisation problem.                         |
| `suggest_fn`       | `Callable`      | —       | **Required.** Batch suggestion function (see signature below).                |
| `n_startup_trials` | `int`           | `8`     | Random trials before `suggest_fn` is called.                                  |
| `q`                | `int`           | `4`     | Batch size — suggestions requested per `suggest_fn` call. Match to `n_jobs`. |
| `seed`             | `int\|None`     | `None`  | Seed for the fallback random sampler.                                         |

#### `suggest_fn` signature

```python
def suggest_fn(
    X: list[list[float]],     # completed trial params, shape (n, d)
    y: list[float],           # raw objective values
    search_space: list[DimSpec],
    q: int,
) -> list[dict[str, Any]]:   # exactly q parameter dicts
    ...
```

`y` values follow the study's direction convention: lower is better for
`direction="minimize"`, higher is better for `direction="maximize"`.
`suggest_fn` is responsible for handling the direction if the underlying
acquisition function requires it.

## Installation

```shell
pip install optuna optunahub

# For the example below:
pip install quantecarlo
```

## Example

This example uses `fantasize_suggest` from the
[quantecarlo](https://pypi.org/project/quantecarlo/) package, a sequential
kriging function that fits a GP in-process and selects a batch of `q`
diverse candidates without any external service.

```python
import optuna
import optunahub
from quantecarlo import fantasize_suggest


module = optunahub.load_module(package="samplers/batch_sampler")
DimSpec = module.DimSpec
BatchSampler = module.BatchSampler

search_space = [
    DimSpec("x", "float", -5.0, 5.0),
    DimSpec("y", "float", -5.0, 5.0),
]

sampler = BatchSampler(
    search_space=search_space,
    suggest_fn=fantasize_suggest,
    q=4,
    n_startup_trials=8,
)

study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(
    lambda trial: trial.suggest_float("x", -5.0, 5.0) ** 2
    + trial.suggest_float("y", -5.0, 5.0) ** 2,
    n_trials=32,
    n_jobs=4,
)
print("Best value:", study.best_value)
print("Best params:", study.best_trial.params)
```

Any callable matching the `suggest_fn` signature can be substituted — a
BoTorch acquisition function, a custom GP, or a wrapper around an external
batch suggestion service.
