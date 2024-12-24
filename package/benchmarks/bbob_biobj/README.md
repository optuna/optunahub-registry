---
author: Optuna team
title: The blackbox optimization benchmarking biobj (bbob-biobj) and biobj-ext (bbob-biobj-ext) test suites
description: A collection of 92 bi-objective benchmark functions. This package is a wrapper of the COCO (COmparing Continuous Optimizers) experiments library.
tags: [benchmark, continuous optimization, multi-objective, BBOB, COCO]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

The bbob-biobj test suite was created by combining existing 55 noiseless single-objective test functions.
bbob-biobj (and its extension, bbob-biobj-ext) has in total of 92 (= original 55 + additional 37) bi-objective functions.
Each benchmark function is provided in dimensions \[2, 3, 5, 10, 20, 40\] with 15 instances.
In this package, all the 92 functions are available.
Please refer to [the paper](https://direct.mit.edu/evco/article/30/2/165/107813/Using-Well-Understood-Single-Objective-Functions) for details about each benchmark function.

## APIs

### class `Problem(function_id: int, dimension: int, instance_id: int = 1)`

- `function_id`: [ID of the bbob benchmark function](https://coco-platform.org/testsuites/bbob-biobj/overview.html) to use. It must be in the range of `[1, 92]`.
- `dimension`: Dimension of the benchmark function. It must be in `[2, 3, 5, 10, 20, 40]`.
- `instance_id`: ID of the instance of the benchmark function. It must be in the range of `[1, 15]`.

#### Methods and Properties

- `search_space`: Return the search space.
  - Returns: `dict[str, optuna.distributions.BaseDistribution]`
- `directions`: Return the optimization directions.
  - Returns: `list[optuna.study.StudyDirection]`
- `__call__(trial: optuna.Trial)`: Evaluate the objective functions and return the objective values.
  - Args:
    - `trial`: Optuna trial object.
  - Returns: `tuple[float, float]`
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


bbob = optunahub.load_module("benchmarks/bbob_biobj")
f92 = bbob.Problem(function_id=92, dimension=40, instance_id=15)

study = optuna.create_study(directions=f92.directions)
study.optimize(f92, n_trials=20)

print(study.best_trials)
```

## Reference

Brockhoff, D., Auger, A., Hansen, N., & Tušar, T. (2022). [Using well-understood single-objective functions in multiobjective black-box optimization test suites. Evolutionary Computation, 30(2)](https://direct.mit.edu/evco/article/30/2/165/107813/Using-Well-Understood-Single-Objective-Functions), 165-193.
