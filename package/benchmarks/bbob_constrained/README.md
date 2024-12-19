---
author: Optuna team
title: The blackbox optimization benchmarking-constrained (bbob-constrained) test suite
description: The bbob-constrained test suite is a suite of 54 non-linearly constrained test functions with varying number of (active and inactive) constraints. This package is a wrapper of the COCO (COmparing Continuous Optimizers) experiments library.
tags: [benchmark, continuous optimization, constrained optimization, BBOB, COCO]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

This package provides a wrapper of the COCO experiments libarary's bbob-constrained test suite.

## APIs

### class `Problem(function_id: int, dimension: int, instance_id: int = 1)`

- `function_id`: [ID of the bbob-constrained benchmark function](https://numbbo.github.io/coco/testsuites/bbob-constrained) to use. It must be in the range of `[1, 54]`.
- `dimension`: Dimension of the benchmark function. It must be in `[2, 3, 5, 10, 20, 40]`.
- `instance_id`: ID of the instance of the benchmark function. It must be in the range of `[1, 15]`.

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
    - `params`: Decision variable like `{"x0": x1_value, "x1": x1_value, ..., "xn": xn_value}`. The number of parameters must be equal to `dimension`.
  - Returns: `float`
- `constraints_func(trial: optuna.Trial.FrozenTrial)`: Evaluate the constraint functions and return the list of constraint functions values.
  - Args:
    - `trial`: Optuna trial object.
  - Returns: `list[float]`
- `evaluate_constraints(params: dict[str, float])`: Evaluate the constraint functions and return the list of constraint functions values.
  - Args:
    - `params`: Decision variable like `{"x0": x1_value, "x1": x1_value, ..., "xn": xn_value}`. The number of parameters must be equal to `dimension`.
  - Returns: `list[float]`

The properties defined by [cocoex.Problem](https://numbbo.github.io/coco-doc/apidocs/cocoex/cocoex.Problem.html) are also available such as `number_of_objectives`.

## Installation

Please install the [coco-experiment](https://github.com/numbbo/coco-experiment/tree/main/build/python) package.

```shell
pip install -U coco-experiment
```

## Example

```python
import optuna
import optunahub


bbob_constrained = optunahub.load_module("benchmarks/bbob_constrained")
constrained_sphere2d = bbob_constrained.Problem(function_id=1, dimension=2, instance_id=1)

study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(
        constraints_func=constrained_sphere2d.constraints_func
    ),
    directions=constrained_sphere2d.directions
)
study.optimize(constrained_sphere2d, n_trials=20)

try:
    print(study.best_trial.params, study.best_trial.value)
except Exception as e:
    print(e)
```

## Details of Benchmark Functions

Please refer to [the paper](https://numbbo.github.io/coco-doc/bbob-constrained/functions.pdf) for details about each benchmark function.

## Reference

Paul Dufossé, Nikolaus Hansen, Dimo Brockhoff, Phillipe R. Sampaio, Asma Atamna, and Anne Auger. [Building scalable test problems for benchmarking constrained optimizers. 2022. To be submitted to the SIAM Journal of Optimization](https://numbbo.github.io/coco-doc/bbob-constrained/functions.pdf).
