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

This sampler currently accepts only `seed`.

## Installation

This sampler requires optional dependencies of Optuna.

```shell
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
