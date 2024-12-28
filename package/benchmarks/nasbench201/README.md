---
author: Optuna Team
title: NATS-Bench (NAS-Bench-201); Benchmarking NAS Algorithms for Architecture Topology and Size
description: The neural architecture search (NAS) benchmark datasets introduced in the paper "NATS-Bench; Benchmarking NAS Algorithms for Architecture Topology and Size"
tags: [benchmark, HPO, NAS, AutoML, hyperparameter optimization, real world problem]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

Neural architecture search benchmark introduced in the paper [`NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size`](https://arxiv.org/abs/2009.00437).
The original benchmark is available [here](https://github.com/D-X-Y/NATS-Bench).
Please note that this benchmark provides the results only at the last epoch of each architecture.

[The preliminary version](https://arxiv.org/abs/2001.00326) is the NAS-Bench-201, but since the widely used name is NAS-Bench-201, we stick to the name, NAS-Bench-201

## APIs

### class `Problem(dataset_id: int, seed: int | None = None, metric_names: list[str] | None = None)`

- `dataset_id`: ID of the dataset to use. It must be in the range of `[0, 2]`. Please use `Problem.available_dataset_names` to see the available dataset names.
- `seed`: The seed for the random number generator of the dataset.
- `metric_names`: The metrics to use in optimization. Defaults to `None`, leading to single-objective optimization of the main metric defined in [here](https://github.com/nabenabe0928/simple-hpo-bench/blob/v0.2.0/hpo_benchmarks/nasbench201.py#L16). Please use `Problem.available_metric_names` to see the available metric names.

#### Methods and Properties

- `search_space`: Return the search space.
  - Returns: `dict[str, optuna.distributions.BaseDistribution]`
- `directions`: Return the optimization directions.
  - Returns: `list[optuna.study.StudyDirection]`
- `metric_names`: The names of the metrics to be used in the optimization.
  - Returns: `list[str]`
- `available_metric_names`: `list[str]`
  - Returns: The names of the available metrics.
- `available_dataset_names`: `list[str]`
  - Returns: The names of the available datasets.
- `__call__(trial: optuna.Trial)`: Evaluate the objective functions and return the objective values.
  - Args:
    - `trial`: Optuna trial object.
  - Returns: `list[float]`
- `evaluate(params: dict[str, int | float | str])`: Evaluate the objective function given a dictionary of parameters.
  - Args:
    - `params`: The parameters defined in `search_space`.
  - Returns: `list[float]`
- `reseed(seed: int | None = None)`: Recreate the random number generator with the given seed.
  - Args:
    - `seed`: The seed to be used.

## Installation

To use this benchmark, you need to install `simple-hpo-bench`.

```shell
$ pip install simple-hpo-bench
```

## Example

```python
from __future__ import annotations

import optuna
import optunahub


hpobench = optunahub.load_module("benchmarks/nasbench201")
problem = hpobench.Problem(dataset_id=0)
study = optuna.create_study()
study.optimize(problem, n_trials=30)
print(study.best_trial)

```

## Others

### Reference

This benchmark was originally introduced by [Xuanyi Dong](https://github.com/D-X-Y), but our backend relies on [`simple-hpo-bench`](https://github.com/nabenabe0928/simple-hpo-bench/).

### Bibtex

```bibtex
@article{dong2021nats,
  title   = {{NATS-Bench}: Benchmarking NAS Algorithms for Architecture Topology and Size},
  author  = {Dong, Xuanyi and Liu, Lu and Musial, Katarzyna and Gabrys, Bogdan},
  doi     = {10.1109/TPAMI.2021.3054824},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year    = {2021},
  note    = {\mbox{doi}:\url{10.1109/TPAMI.2021.3054824}}
}
@inproceedings{dong2020nasbench201,
  title     = {{NAS-Bench-201}: Extending the Scope of Reproducible Neural Architecture Search},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  url       = {https://openreview.net/forum?id=HJxyZkBKDr},
  year      = {2020}
}
```
