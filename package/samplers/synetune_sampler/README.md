---
author: Syne Tune
title: 'Syne Tune: Large-Scale and Reproducible Hyperparameter Optimization'
description: Syne Tune provides state-of-the-art algorithms for hyperparameter optimization (HPO).
tags: [sampler, Bayesian optimization, CQR, Bore]
optuna_versions: [4.2.1]
license: MIT License
---

## APIs

A sampler that uses Syne Tune v0.14.1 that can be run by the following:

```shell
$ pip install optunahub syne-tune
```

Please check the API reference for more details:

- https://syne-tune.readthedocs.io/en/latest/\_apidoc/modules.html

### `SyneTuneSampler(metric: str, search_space: dict[str, BaseDistribution], direction: str = "minimize", searcher_method: str = "CQR", searcher_kwargs: dict = None)`

- `search_space`: A dictionary of Optuna distributions.
- `direction`: Defines direction of optimization. Must be one of the following: `[minimize, maximize]`.
- `metric`: The metric to be optimized.
- `searcher_method`: The optimization method to be run on the objective. Currently supported searcher methods: `[CQR, REA, BORE, RandomSearch]`.
  - **RandomSearch**: Selects hyperparameters randomly from the search space, providing a simple baseline method that requires no prior knowledge.
  - **BORE**: Bayesian Optimization with Density-Ratio Estimation, an adaptive method that models the probability of improvement using density estimation.
  - **REA**: Regularized Evolution Algorithm, a population-based evolutionary approach that mutates and selects the best-performing hyperparameter sets over time.
  - **CQR**: Conformal Quantile Regression, a robust uncertainty-aware optimization method that uses quantile regression for reliable performance estimation.
- `searcher_kwargs`: Optional. Additional arguments for the searcher_method. More details can be found in the API documentation

## Installation

```bash
pip install -r https://hub.optuna.org/samplers/synetune_sampler/requirements.txt
```

## Example

```python
import optuna
import optunahub


SyneTuneSampler = optunahub.load_module("samplers/synetune_sampler").SyneTuneSampler


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    return x**2 + y**2


n_trials = 100
sampler = SyneTuneSampler(
    search_space={
        "x": optuna.distributions.FloatDistribution(-10, 10),
        "y": optuna.distributions.IntDistribution(-10, 10),
    },
    searcher_method="CQR",
    metric="mean_loss",
)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=n_trials)
print(study.best_trial.params)
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/synetune_sampler/example.py) for a full example.

## Others

Syne Tune is maintained by the Syne Tune team. If you have trouble using Syne Tune, a concrete question or found a bug, please create an issue under the [Syne Tune](https://github.com/syne-tune/syne-tune) repository.

### Reference

```bibtex
Salinas, D., Seeger, M., Klein, A., Perrone, V., Wistuba, M., & Archambeau, C. (2022, September). Syne tune: A library for large scale hyperparameter tuning and reproducible research. In International Conference on Automated Machine Learning (pp. 16-1). PMLR.
```
