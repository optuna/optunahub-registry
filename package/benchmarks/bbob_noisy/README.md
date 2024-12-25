---
author: Optuna team
title: The blackbox optimization benchmarking noisy (bbob-noisy) test suite
description: The blackbox optimization benchmarking noisy (bbob-noisy) test suite consists of 30 noisy single-objective test functions. This package is a wrapper of the COCO (COmparing Continuous Optimizers) experiments library.
tags: [benchmark, continuous optimization, BBOB, COCO]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

The blackbox optimization benchmarking noisy (bbob-noisy) test suite comprises 30 noisy test functions. Each benchmark function is provided in dimensions \[2, 3, 5, 10, 20, 40\] with 15 instances. Please refer to [the paper](https://inria.hal.science/inria-00369466v1) for details about each benchmark function.

## APIs

### class `Problem(function_id: int, dimension: int, instance_id: int = 1)`

- `function_id`: [ID of the bbob benchmark function](https://numbbo.github.io/coco/testsuites/bbob-noisy) to use. It must be in the range of `[101, 130]`.
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
- `evaluate(params: dict[str, float])`: Evaluate the objective function given a dictionary of parameters.
  - Args:
    - `params`: Decision variable like `{"x0": x1_value, "x1": x1_value, ..., "xn": xn_value}`. The number of parameters must be equal to `dimension`.
  - Returns: `float`

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


bbob_noisy = optunahub.load_module("benchmarks/bbob_noisy")
sphere2d_gaussian = bbob_noisy.Problem(function_id=101, dimension=2, instance_id=1)

study = optuna.create_study(directions=sphere2d_gaussian.directions)
study.optimize(sphere2d_gaussian, n_trials=20)

print(study.best_trial.params, study.best_trial.value)
```

## Reference

Nikolaus Hansen, Steffen Finck, Raymond Ros, Anne Auger. [Real-Parameter Black-Box Optimization Benchmarking 2009: Noisy Functions Definitions. \[Research Report\] RR-6869, INRIA. 2009. ⟨inria-00369466⟩](https://inria.hal.science/inria-00369466v1)
