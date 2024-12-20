---
author: Optuna team
title: The WFG tookit implementation
description: The multi-objective optimization benchmarking toolkit called WFG. It provides a set of scalable test problems for continuous multi-objective optimization.
tags: [benchmark, continuous optimization, multi-objective optimization]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

The WFG toolkit provides a set of scalable test problems for continuous multi-objective optimization. It is designed to provide a range of test problems that represent different characteristics of multi-objective optimization problems. The toolkit is designed to be scalable, with the number of objectives and the number of decision variables being easily configurable. The toolkit is also designed to be flexible, with the shape of the Pareto front and the distribution of the Pareto-optimal solutions being easily configurable.

## APIs

### class `Problem(problem_id: int, dimension: int, n_objectives: int, k: int)`

- `problem_id`: The problem ID.
- `dimension`: The number of decision variables.
- `n_objectives`: The number of objectives.
- `k`: The degree of the Pareto front curvature.

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
  - Returns: `list[float]`

## Example

```python
import optuna
import optunahub


wfg = optunahub.load_module("benchmarks/wfg")
wfg1 = wfg.Problem(problem_id=1, dimension=3, n_objectives=2, k=1)

study = optuna.create_study(directions=wfg1.directions)
study.optimize(wfg1, n_trials=20)

best_trials = study.best_trials
for trial in best_trials:
    print(trial.values)
```

## Reference

S. Huband, P. Hingston, L. Barone, and L. While, A review of multiobjective test problems and a scalable test problem toolkit, IEEE Transactions on Evolutionary Computation, 2006, 10(5): 477-506. (https://ro.ecu.edu.au/cgi/viewcontent.cgi?article=3021&context=ecuworks)
