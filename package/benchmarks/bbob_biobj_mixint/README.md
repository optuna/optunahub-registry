---
author: Optuna team
title: The blackbox optimization benchmarking biobj-mixed-integer (biobj-bbob-mixint) test suite
description: The blackbox optimization benchmarking mixed-integer (biobj-bbob-mixint) test suite consists of 92 noiseless mixed-integer bi-objective test functions. This package is a wrapper of the COCO (COmparing Continuous Optimizers) experiments library.
tags: [benchmark, mixed-integer, multi-objective, BBOB, COCO]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

The blackbox optimization benchmarking biobj-mixed-integer (bbob-biobj-mixint) test suite comprises 92 noiseless mixed-integer bi-objective test functions. Each benchmark function is provided in dimensions \[5, 10, 20, 40, 80, 160\] with 15 instances. Please refer to [the paper](https://dl.acm.org/doi/abs/10.1145/3321707.3321868) for details about each benchmark function.

## APIs

### class `Problem(function_id: int, dimension: int, instance_id: int = 1)`

- `function_id`: [ID of the bbob benchmark function](https://numbbo.github.io/coco/testsuites/bbob-biobj-mixint) to use. It must be in the range of `[1, 92]`.
- `dimension`: Dimension of the benchmark function. It must be in `[5, 10, 20, 40, 80, 160]`.
- `instance_id`: ID of the instance of the benchmark function. It must be in the range of `[1, 15]`.

#### Methods and Properties

- `search_space`: Return the search space.
  - Returns: `dict[str, optuna.distributions.BaseDistribution]`
- `directions`: Return the optimization directions.
  - Returns: `list[optuna.study.StudyDirection]`
- `__call__(trial: optuna.Trial)`: Evaluate the objective functions and return the objective value.
  - Args:
    - `trial`: Optuna trial object.
  - Returns: `tupe[float, float]`
- `evaluate(params: dict[str, float])`: Evaluate the objective functions given a dictionary of parameters.
  - Args:
    - `params`: Decision variable like `{"x0": x1_value, "x1": x1_value, ..., "xn": xn_value}`. The number of parameters must be equal to `dimension`.
  - Returns: `tuple[float, float]`

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


bbob = optunahub.load_module("benchmarks/bbob_biobj_mixint")
f92 = bbob.Problem(function_id=92, dimension=40, instance_id=15)

study = optuna.create_study(directions=f92.directions)
study.optimize(f92, n_trials=20)

print(study.best_trials)
```

## Reference

Tušar, T., Brockhoff, D., & Hansen, N. (2019, July). [Mixed-integer benchmark problems for single-and bi-objective optimization](https://dl.acm.org/doi/abs/10.1145/3321707.3321868). In Proceedings of the Genetic and Evolutionary Computation Conference (pp. 718-726).
