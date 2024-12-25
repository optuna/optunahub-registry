---
author: Optuna team
title: The ZDT Problem Collection
description: The ZDT Problem Collection (Zitzler et al. 2000) is a widely-used benchmark suite for multi-objective optimization. This package is a wrapper of the optproblems library.
tags: [benchmark, multi-objective, ZDT, optproblems]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

This package provides a wrapper of the [optproblems](https://www.simonwessing.de/optproblems/doc/index.html) library's ZDT test suite, which consists of 6 kinds of benchmark problems. For the details of the benchmark problems, please take a look at the original paper (Zitzler et al., 2000) in the reference section.

## APIs

### class `Problem(function_id: int, **kwargs: Any)`

- `function_id`: Function ID of the WFG problem in \[1, 9\].
- `kwargs`: Arbitrary keyword arguments, please refer to [the optproblems documentation](https://www.simonwessing.de/optproblems/doc/zdt.html) for more details.

#### Methods and Properties

- `search_space`: Return the search space.
  - Returns: `dict[str, optuna.distributions.BaseDistribution]`
- `directions`: Return the optimization directions.
  - Returns: `list[optuna.study.StudyDirection]`
- `__call__(trial: optuna.Trial)`: Evaluate the objective functions and return the objective values.
  - Args:
    - `trial`: Optuna trial object.
  - Returns: `float`
- `evaluate(params: dict[str, float])`: Evaluate the objective functions and return the objective values.
  - Args:
    - `params`: Decision variable like `{"x0": x1_value, "x1": x1_value, ..., "xn": xn_value}`. The number of parameters must be equal to `num_variables`.
  - Returns: `float`

The properties defined by [optproblems](https://www.simonwessing.de/optproblems/doc/zdt.html) are also available.

## Installation

Please install the [optproblems](https://pypi.org/project/optproblems/) package.

```shell
pip install -U optproblems
```

## Example

```python
import optuna
import optunahub


zdt = optunahub.load_module("benchmarks/zdt")
zdt4 = zdt.Problem(function_id=4)

study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=42),
    directions=zdt4.directions,
)
study.optimize(zdt4, n_trials=100)
optuna.visualization.plot_pareto_front(study).show()
```

## Reference

Zitzler, E., Deb, K., & Thiele, L. (2000). [Comparison of multiobjective evolutionary algorithms: Empirical results](https://ieeexplore.ieee.org/abstract/document/6787994). Evolutionary computation, 8(2), 173-195.
