---
author: Difan Deng
title: SMAC3
description: SMAC offers a robust and flexible framework for Bayesian Optimization to support users in determining well-performing hyperparameter configurations for their (Machine Learning) algorithms, datasets and applications at hand. The main core consists of Bayesian Optimization in combination with an aggressive racing mechanism to efficiently decide which of two configurations performs better.
tags: [sampler, Bayesian optimization, Gaussian process, Random Forest]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- SAMCSampler

## Installation

```bash
pip install -r https://hub.optuna.org/samplers/smac_sampler/requirements.txt
```

## Example

```python
import optuna
import optunahub


module = optunahub.load_module("samplers/smac_sampler")
SMACSampler = module.SMACSampler


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    return x**2 + y**2


n_trials = 100
sampler = SMACSampler(
    {
        "x": optuna.distributions.FloatDistribution(-10, 10),
        "y": optuna.distributions.IntDistribution(-10, 10),
    },
    n_trials=n_trials,
)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=n_trials)
print(study.best_trial.params)
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/smac_sampler/example.py) for a full example.
![History Plot](images/smac_sampler_history.png "History Plot")

## Others

SMAC is maintained by the SMAC team in [automl.org](https://www.automl.org/). If you have trouble using SMAC, a concrete question or found a bug, please create an issue under the [SMAC](https://github.com/automl/SMAC3) repository.

For all other inquiries, please write an email to smac\[at\]ai\[dot\]uni\[dash\]hannover\[dot\]de.

### Reference

Lindauer et al. "SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization", Journal of Machine Learning Research, http://jmlr.org/papers/v23/21-0888.html
