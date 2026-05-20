---
author: Mark Shipman
title: Remote GP q-EI Sampler
description: Batch Bayesian optimisation via q-Expected Improvement, delegating GP fitting and candidate scoring to a remote HTTP service.
tags: [sampler, bayesian-optimization, batch]
optuna_versions: [4.8.0]
license: MIT License
---

## Abstract

`qEISampler` is a batch Bayesian optimisation sampler for Optuna.
Instead of fitting a Gaussian Process locally, it sends the current observations to a user-supplied HTTP endpoint that returns a batch of `q` candidates maximising the q-Expected Improvement (q-EI) acquisition function.

This design is useful when:

- GP fitting is too slow to run inside the Optuna worker process (large datasets, expensive kernels).
- You want to centralise the surrogate model on a GPU server or managed service (e.g. Modal, AWS Lambda, Cloud Run).
- You need to reuse the same GP service across multiple concurrent Optuna studies.

During the startup phase (fewer than `n_startup_trials` complete trials) the sampler falls back to random search automatically.

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

### `qEISampler(search_space, api_url, ...)`

| Argument           | Type            | Default        | Description                                                               |
| ------------------ | --------------- | -------------- | ------------------------------------------------------------------------- |
| `search_space`     | `list[DimSpec]` | —              | **Required.** Dimensions of the optimisation problem.                     |
| `api_url`          | `str`           | —              | **Required.** URL of the GP suggestion endpoint (see API contract below). |
| `n_startup_trials` | `int`           | `8`            | Random trials before GP is used.                                          |
| `q`                | `int`           | `4`            | Batch size — number of candidates requested per API call.                 |
| `n_candidates`     | `int`           | `512`          | Quasi-random candidates evaluated by the acquisition function.            |
| `train_steps`      | `int`           | `60`           | GP hyperparameter optimisation steps on the server side.                  |
| `lr`               | `float`         | `0.1`          | Learning rate for GP hyperparameter optimisation.                         |
| `xi`               | `float`         | `0.01`         | Exploration bonus added to the best observed value before computing EI.   |
| `mode`             | `str`           | `"production"` | `"debug"` prints per-batch EI scores to stdout.                           |
| `seed`             | `int\|None`     | `None`         | Seed for the fallback random sampler.                                     |
| `timeout`          | `float`         | `120.0`        | HTTP request timeout in seconds.                                          |

## Backend API contract

The sampler POSTs JSON to `api_url` and expects a JSON response.

**Request body**

```json
{
  "X": [[x1_dim1, x1_dim2, ...], [x2_dim1, ...], ...],
  "y": [-val1, -val2, ...],
  "search_space": [{"name": "x", "type": "float", "low": -5, "high": 5, "log": false, "step": null}],
  "q": 4,
  "n_candidates": 512,
  "train_steps": 60,
  "lr": 0.1,
  "xi": 0.01,
  "mode": "production"
}
```

Notes:

- `X` rows correspond to completed trials; columns correspond to `search_space` dims in order.
- `y` values are **negated** trial objectives (the server maximises q-EI; Optuna minimises).

**Response body**

```json
{
  "candidates": [
    {"x": [v1_dim1, v1_dim2, ...]},
    {"x": [v2_dim1, v2_dim2, ...]},
    ...
  ]
}
```

The server must return exactly `q` candidates.
Each `x` array must have the same length as `search_space`.

Optional debug fields (`ei_all`, `ei_scores`) are consumed when `mode="debug"`.

## Installation

```shell
pip install optuna optunahub
```

No additional Python dependencies are required — the sampler uses only the standard library and Optuna.

### Backend endpoint

You must supply an `api_url` that implements the contract above. Two options:

**Use the hosted endpoint** (no setup required):

```
https://markshipman4273--bo-gp-service-gp-suggest.modal.run
```

This is a publicly available Modal deployment. Pass it directly as `api_url`.

**Deploy your own** using the open-source backend at
[sign-of-fourier/quantecarlo](https://github.com/sign-of-fourier/quantecarlo),
or implement the request/response contract on any HTTP server.

## Example

```python
import optuna
import optunahub


module = optunahub.load_module(package="samplers/q_ei_sampler")
DimSpec = module.DimSpec
qEISampler = module.qEISampler

# Substitute the URL of your own GP service.
sampler = qEISampler(
    search_space=[
        DimSpec("lr", "float", 1e-4, 1e-1, log=True),
        DimSpec("n_hidden", "int", 16, 256),
    ],
    api_url="https://your-gp-service/suggest",
    q=4,
    n_startup_trials=8,
)

study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(lambda trial: trial.suggest_float("lr", 1e-4, 1e-1, log=True) ** 2, n_trials=40)
print("Best value:", study.best_value)
```
