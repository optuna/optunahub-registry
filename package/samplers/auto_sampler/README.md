---
author: Optuna Team
title: AutoSampler
description: This sampler automatically chooses an appropriate built-in sampler for the provided objective function.
tags: [sampler, automatic, automated]
optuna_versions: [4.0.0]
license: MIT License
---

## Abstract

This package provides a sampler based on Gaussian process-based Bayesian optimization. The sampler is highly sample-efficient, so it is suitable for computationally expensive optimization problems with a limited evaluation budget, such as hyperparameter optimization of machine learning algorithms.

## Class or Function Names

- AutoSampler

This sampler currently accepts only `seed` and `constraints_func`.
`constraints_func` enables users to handle constraints along with the objective function.
This argument follows the same convention as the other samplers, so please take a look at [the reference](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).

## Installation

This sampler requires optional dependencies of Optuna.

```shell
# TODO
$ pip install "optuna[optional]"
```

## Example

```python
import optunahub


def objective(trial):
  x = trial.suggest_float("x", -5, 5)
  return x**2


module = optunahub.load_module(package="samplers/auto_sampler")
study = optuna.create_study(sampler=module.AutoSampler())
study.optimize(objective, n_trials=300)
```

### Test

To execute the tests for `AutoSampler`, please run the following commands. The test file is provided in the package.

```sh
pip install pytest
```

```python
python -m pytest package/samplers/tests/test_auto_sampler.py
```
