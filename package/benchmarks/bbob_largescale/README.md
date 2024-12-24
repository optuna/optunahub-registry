---
author: Optuna team
title: The blackbox optimization benchmarking largescale (bbob-largescale) test suite
description: The blackbox optimization benchmarking largescale (bbob-largescale) test suite consists of high-dimensional 24 noiseless single-objective test functions including Sphere, Ellipsoidal, Rastrigin, Rosenbrock, etc. This package is a wrapper of the COCO (COmparing Continuous Optimizers) experiments library.
tags: [benchmark, continuous optimization, high-dimensional, BBOB, COCO]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

The blackbox optimization benchmarking largescale (bbob-largescale) test suite comprises high-dimensional 24 noiseless single-objective test functions. Each benchmark function is provided in dimensions \[20, 40, 80, 160, 320, 640\] with 15 instances. Please refer to [the paper](https://arxiv.org/abs/1903.06396) for details about each benchmark function.

## APIs

### class `Problem(function_id: int, dimension: int, instance_id: int = 1)`

- `function_id`: [ID of the bbob benchmark function](https://numbbo.github.io/coco/testsuites/bbob-largescale) to use. It must be in the range of `[1, 24]`.
- `dimension`: Dimension of the benchmark function. It must be in `[20, 40, 80, 160, 320, 640]`.
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


bbob = optunahub.load_module("benchmarks/bbob-largescale")
sphere640d = bbob.Problem(function_id=1, dimension=640, instance_id=1)

study = optuna.create_study(directions=sphere640d.directions)
study.optimize(sphere640d, n_trials=20)

print(study.best_trial.params, study.best_trial.value)
```

## Reference

Elhara, O., Varelas, K., Nguyen, D., Tusar, T., Brockhoff, D., Hansen, N., & Auger, A. (2019). [COCO: the large scale black-box optimization benchmarking (BBOB-largescale) test suite](https://arxiv.org/abs/1903.06396). arXiv preprint arXiv:1903.06396.
