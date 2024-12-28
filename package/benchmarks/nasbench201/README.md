---
author: Optuna Team
title: HPOLib; Tabular Benchmarks for Hyperparameter Optimization and Neural Architecture Search
description: The hyperparameter optimization benchmark datasets introduced in the paper "Tabular Benchmarks for Hyperparameter Optimization and Neural Architecture Search"
tags: [benchmark, HPO, AutoML, hyperparameter optimization, real world problem]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

Hyperparameter optimization benchmark introduced in the paper "Tabular Benchmarks for Hyperparameter Optimization and Neural Architecture Search".

## APIs

Please provide API documentation describing how to use your package's functionalities.
The documentation format is arbitrary, but at least the important class/function names that you implemented should be listed here.
More users will take advantage of your package by providing detailed and helpful documentation.

**Example**

- `MoCmaSampler(*, search_space: dict[str, BaseDistribution] | None = None, popsize: int | None = None, seed: int | None = None)`
  - `search_space`: A dictionary containing the search space that defines the parameter space. The keys are the parameter names and the values are [the parameter's distribution](https://optuna.readthedocs.io/en/stable/reference/distributions.html). If the search space is not provided, the sampler will infer the search space dynamically.
    Example:
    ```python
    search_space = {
        "x": optuna.distributions.FloatDistribution(-5, 5),
        "y": optuna.distributions.FloatDistribution(-5, 5),
    }
    MoCmaSampler(search_space=search_space)
    ```
  - `popsize`: Population size of the CMA-ES algorithm. If not provided, the population size will be set based on the search space dimensionality. If you have a sufficient evaluation budget, it is recommended to increase the popsize.
  - `seed`: Seed for random number generator.

Note that because of the limitation of the algorithm, only non-conditional numerical parameters can be sampled by the MO-CMA-ES algorithm, and categorical and conditional parameters are handled by random search.

## Installation

```shell
$ pip install simple-hpo-bench
```

## Example

```python
import optuna
import optunahub


nasbench201 = optunahub.load_module("benchmarks/nasbench201")
constrained_sphere2d = nasbench201.Problem(function_id=1, dimension=2, instance_id=1)

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

## Others

### Reference

This benchmark was originally introduced by [AutoML.org](https://github.com/automl/nas_benchmarks/tree/master), but our backend relies on [`simple-hpo-bench`](https://github.com/nabenabe0928/simple-hpo-bench/).

### Bibtex

```bibtex
@article{klein2019tabular,
  title={Tabular benchmarks for joint architecture and hyperparameter optimization},
  author={Klein, Aaron and Hutter, Frank},
  journal={arXiv preprint arXiv:1905.04970},
  year={2019}
}
```
