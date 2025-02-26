---
author: Difan Deng
title: SMAC3
description: SMAC offers a robust and flexible framework for Bayesian Optimization to support users in determining well-performing hyperparameter configurations for their (Machine Learning) algorithms, datasets and applications at hand. The main core consists of Bayesian Optimization in combination with an aggressive racing mechanism to efficiently decide which of two configurations performs better.
tags: [sampler, Bayesian optimization, Gaussian process, Random Forest]
optuna_versions: [3.6.1]
license: MIT License
---

## APIs

A sampler that uses SMAC3 v2.2.0 verified by unittests that can be run by the following:

```shell
$ pip install pytest optunahub syne-tune
$ python -m pytest package/samplers/synetune_sampler/tests/
```

Please check the API reference for more details:

- https://syne-tune.readthedocs.io/en/latest/\_apidoc/modules.html

### `SyneTuneSampler(search_space: dict[str, BaseDistribution],mode: str = "min",metric: str = "mean_loss",searcher_method: str = "random_search",searcher_kwargs: dict = None)`

- `search_space`: A dictionary of Optuna distributions.
- `mode`: Defines direction of optimization. Must be one of the following: `[min, max]`.
- `metric`: The metric to be optimized.
- `searcher_method`: The optimization method to be run on the objective. Currently supports: `[cqr, kde, regularized_evolution, bore]`.
- `searcher_kwargs`: Optional. Additional arguments for the searcher_method. More details can be found in the API documentation

## Installation

```bash
pip install -r https://hub.optuna.org/samplers/synetune_sampler/requirements.txt
```

## Example

```python
import optuna
import optunahub


module = optunahub.load_module("samplers/synetune_sampler")
SyneTuneSampler = module.SyneTuneSampler


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    return x**2 + y**2


n_trials = 100
sampler = SyneTuneSampler(
    {
        "x": optuna.distributions.FloatDistribution(-10, 10),
        "y": optuna.distributions.IntDistribution(-10, 10),
    },
    searcher_method="cqr",
    metric="mean_loss",
)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=n_trials)
print(study.best_trial.params)
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/smac_sampler/example.py) for a full example.

## Others

SyneTune is maintained by the SyneTune team. If you have trouble using SyneTune, a concrete question or found a bug, please create an issue under the [Synetune](https://github.com/syne-tune/syne-tune) repository.

For all other inquiries, please file an issue on the SyneTune repository.

### Reference

Salinas, D., Seeger, M., Klein, A., Perrone, V., Wistuba, M., & Archambeau, C. (2022, September). Syne tune: A library for large scale hyperparameter tuning and reproducible research. In International Conference on Automated Machine Learning (pp. 16-1). PMLR.
