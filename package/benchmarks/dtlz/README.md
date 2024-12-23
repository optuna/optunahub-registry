---
author: Optuna team
title: The DTLZ Problem Collection
description: The DTLZ Problem Collection (Deb et al. 2001) is a widely-used benchmark suite for multi-objective optimization. This package is a wrapper of the optproblems library.
tags: [benchmark, continuous optimization, multi-objective, DTLZ, optproblems]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

This package provides a wrapper of the [optproblems](https://www.simonwessing.de/optproblems/doc/index.html) library's DTLZ test suite, which consists of 7 kinds of continuous problems with variadic objectives and variables. For the details of the benchmark problems, please take a look at the original paper (Deb et al., 2001) in the reference section.

## APIs

### class `Problem(function_id: int, num_objectives: int, num_variables: int, k: int, **kwargs: Any)`

- `function_id`: Function ID of the WFG problem in \[1, 9\].
- `num_objectives`: Number of objectives.
- `num_variables`: Number of variables.
- `kwargs`: Arbitrary keyword arguments, please refer to [the optproblems documentation](https://www.simonwessing.de/optproblems/doc/dtlz.html) for more details.

#### Methods and Properties

- `search_space`: Return the search space.
  - Returns: `dict[str, optuna.distributions.BaseDistribution]`
- `directions`: Return the optimization directions.
  - Returns: `list[optuna.study.StudyDirection]`
- `__call__(trial: optuna.Trial)`: Evaluate the objective function and return the objective value.
  - Args:
    - `trial`: Optuna trial object.
  - Returns: `float`
- `evaluate(params: dict[str, float])`: Evaluate the objective function and return the objective value.
  - Args:
    - `params`: Decision variable like `{"x0": x1_value, "x1": x1_value, ..., "xn": xn_value}`. The number of parameters must be equal to `num_variables`.
  - Returns: `float`

The properties defined by [optproblems](https://www.simonwessing.de/optproblems/doc/dtlz.html) are also available such as `get_optimal_solutions`.

## Installation

Please install the [optproblems](https://pypi.org/project/optproblems/) package.

```shell
pip install -U optproblems
```

## Example

```python
import optuna
import optunahub


dtlz = optunahub.load_local_module("benchmarks/dtlz", registry_root="../../")
dtlz2 = dtlz.Problem(function_id=2, num_objectives=2, num_variables=3)

study_tpe = optuna.create_study(
    study_name="TPESampler",
    sampler=optuna.samplers.TPESampler(seed=42),
    directions=dtlz2.directions,
)
study_tpe.optimize(dtlz2, n_trials=1000)
optuna.visualization.plot_pareto_front(study_tpe).show()
```

## Reference

Deb, K., Thiele, L., Laumanns, M., & Zitzler, E. (2001). [Scalable Test Problems for Evolutionary Multi-Objective Optimization](https://www.research-collection.ethz.ch/handle/20.500.11850/145762).
