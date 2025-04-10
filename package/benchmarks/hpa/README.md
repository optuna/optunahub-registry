---
author: Optuna team
title: Single and Multi-objective Optimization Benchmark Problems Focusing on Human-Powered Aircraft Design
description: The benchmark problem for human-powered aircraft design introduced in the paper `Single and Multi-Objective Optimization Benchmark Problems Focusing on Human-Powered Aircraft Design`
tags: [benchmark, HPA, multi-objective, human-powered aircraft]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

The benchmark for human-powered aircraft (hpa) design is introduced in the paper [Single and Multi-Objective Optimization Benchmark Problems Focusing on Human-Powered Aircraft Design](https://arxiv.org/abs/2312.08953).
The original benchmark is available [here](https://github.com/Nobuo-Namura/hpa).
This package serves as a wrapper for the original benchmark.

## APIs

### class `ConstrainedProblem(problem_name: str, n_div: int = 4, level: int = 0, NORMALIZED: bool = True )`

- `problem_name`: The name of a benchmark problem. All problem names and their explanations are provided [here](https://github.com/Nobuo-Namura/hpa?tab=readme-ov-file#benchmark-problem-definition).
- `n_div`: The wing segmentation number and alters the problem's dimension. It must be an integer greater than 0.
- `level`: The difficulty level of the problem. It must be in `[0, 1, 2]`.
- `NORMALIZED`: Whether to use normalized design variables or not.

#### Method and Properties

- `search_space`: Return the search space.
  - Returns: `dict[str, optuna.distributions.BaseDistribution]`
- `directions`: Return the optimization directions.
  - Returns: `list[optuna.study.StudyDirection]`
- `evaluate(params: dict[str, float])`: Evaluate the objective function given a dictionary of parameters.
  - Args:
    - `params`: A dictionary representing decision variable like `{"x0": x1_value, "x1": x1_value, ..., "xn": xn_value}`. The number of parameters must be equal to `self.nx`. `xn_value` must be a `float` in `[0, 1]`.
  - Returns: List of length `self.nf`.
- `evaluate_constraints(params: dict[str, float])`: Evaluate the constraint functions and return the list of constraint functions values.
  - Args:
    - `params`: A dictionary representing the decision variables, with the same format and value range as in evaluate.
  - Returns: List of length `self.ng`. If `self.ng == 0` (means that this is not a constrained problem), this function raises `TypeError`.

The properties and functions of classes in [`hpa.problem`](https://github.com/Nobuo-Namura/hpa/blob/main/hpa/problem.py) are also available such as `nx`, `plot_wing3d`.

## Installation

Please install [hpa repository](https://github.com/Nobuo-Namura/hpa) in your python environment.

```
pip install git+https://github.com/Nobuo-Namura/hpa
```

or clone the repository and get into it, and run

```
pip install .
```

See the installation details by visiting [here](https://github.com/Nobuo-Namura/hpa?tab=readme-ov-file#installation).

## Example

```Python
from __future__ import annotations

import optuna
import optunahub


hpa = optunahub.load_module("benchmarks/hpa")
problem = hpa.ConstrainedProblem("HPA131") 
study = optuna.create_study(directions=problem.directions)
study.optimize(problem, n_trials=10)
print(study.best_trial)
```

## Reference

Namura, N. [Single and Multi-objective Optimization Benchmark Problems Focusing on Human-Powered Aircraft Design](https://link.springer.com/chapter/10.1007/978-981-96-3506-1_14).
