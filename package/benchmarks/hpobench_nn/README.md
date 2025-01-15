---
author: Optuna Team
title: HPOBench; A Collection of Reproducible Multi-Fidelity Benchmark Problems for HPO
description: The hyperparameter optimization benchmark datasets introduced in the paper "HPOBench; A Collection of Reproducible Multi-Fidelity Benchmark Problems for HPO"
tags: [benchmark, HPO, NAS, AutoML, hyperparameter optimization, real world problem]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

Hyperparameter optimization benchmark introduced in the paper [`HPOBench: A Collection of Reproducible Multi-Fidelity Benchmark Problems for HPO`](https://arxiv.org/abs/2109.06716).
The original benchmark is available [here](https://github.com/automl/hpobench).
Please note that this benchmark provides the results only at the last epoch of each configuration.

## APIs

### class `Problem(dataset_id: int, seed: int | None = None, metric_names: list[str] | None = None)`

- `dataset_id`: ID of the dataset to use. It must be in the range of `[0, 7]`. Please use `Problem.available_dataset_names` to see the available dataset names.
- `seed`: The seed for the random number generator of the dataset.
- `metric_names`: The metrics to use in optimization. Defaults to `None`, leading to single-objective optimization of the main metric defined in [here](https://github.com/nabenabe0928/simple-hpo-bench/blob/v0.2.0/hpo_benchmarks/hpolib.py#L16). Please use `Problem.available_metric_names` to see the available metric names.

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


hpobench = optunahub.load_module("benchmarks/hpobench_nn")
problem = hpobench.Problem(dataset_id=0)
study = optuna.create_study()
study.optimize(problem, n_trials=30)
print(study.best_trial)

```

## Others

### Reference

This benchmark was originally introduced by [AutoML.org](https://github.com/automl/hpobench), but our backend relies on [`simple-hpo-bench`](https://github.com/nabenabe0928/simple-hpo-bench/).

### Bibtex

```bibtex
@inproceedings{
  eggensperger2021hpobench,
  title={{HPOB}ench: A Collection of Reproducible Multi-Fidelity Benchmark Problems for {HPO}},
  author={Katharina Eggensperger and Philipp M{\"u}ller and Neeratyoy Mallik and Matthias Feurer and Rene Sass and Aaron Klein and Noor Awad and Marius Lindauer and Frank Hutter},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021},
  url={https://openreview.net/forum?id=1k4rJYEwda-}
}
```
