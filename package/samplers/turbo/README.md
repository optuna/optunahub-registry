---
author: Optuna Team
title: TuRBOSampler
description: This sampler performs Bayesian optimization in adaptive trust regions using Gaussian Processes
tags: [sampler, Bayesian optimization]
optuna_versions: [4.6.0]
license: MIT License
---

## Abstract

TuRBOSampler implements Bayesian optimization with trust regions. It places local trust regions around the current best solutions and fits Gaussian Process (GP) models within those regions. Operating within adaptive local regions reduces high-dimensional sample complexity, yielding accurate fits with fewer trials.

Please refer to the paper, [Scalable Global Optimization via Local Bayesian Optimization](https://proceedings.neurips.cc/paper_files/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf) for more information.

## APIs

- `TuRBOSampler(*, n_startup_trials: int = 4, n_trust_region: int = 5, success_tolerance: int = 3, failure_tolerance: int = 5, seed: int | None = None, independent_sampler: BaseSampler | None = None, deterministic_objective: bool = False, warn_independent_sampling: bool = True)`
  - `n_startup_trials`: Number of initial trials PER TRUST REGION. Default is 2. As suggested in the original paper, consider setting this to 2\*(number of parameters).
  - `n_trust_region`: Number of trust regions. Default is 5.
  - `success_tolerance`: Number of consecutive successful iterations required to expand the trust region. Default is 3.
  - `failure_tolerance`: Number of consecutive failed iterations required to shrink the trust region. Default is 5. As suggested in the original paper, consider setting this to max(5, number of parameters).
  - `seed`: Random seed to initialize internal random number generator. Defaults to :obj:`None` (a seed is picked randomly).
  - `independent_sampler`: Sampler used for initial sampling (for the first `n_startup_trials` trials) and for conditional parameters. Defaults to :obj:`None` (a random sampler with the same `seed` is used).
  - `deterministic_objective`: Whether the objective function is deterministic or not. If :obj:`True`, the sampler will fix the noise variance of the surrogate model to the minimum value (slightly above 0 to ensure numerical stability). Defaults to :obj:`False`. Currently, all the objectives will be assume to be deterministic if :obj:`True`.
  - `warn_independent_sampling`: If this is :obj:`True`, a warning message is emitted when the value of a parameter is sampled by using an independent sampler, meaning that no GP model is used in the sampling. Note that the parameters of the first trial in a study are always sampled via an independent sampler, so no warning messages are emitted in this case.

Note that categorical parameters are currently unsupported, and multi-objective optimization is not available.

## Installation

```shell
$ pip install torch scipy
```

## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


sampler = optunahub.load_module(package="samplers/turbo").TuRBOSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=200)

```

## Others

### Bibtex

```
@inproceedings{eriksson2019scalable,
  title = {Scalable Global Optimization via Local {Bayesian} Optimization},
  author = {Eriksson, David and Pearce, Michael and Gardner, Jacob and Turner, Ryan D and Poloczek, Matthias},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {5496--5507},
  year = {2019},
  url = {http://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization.pdf},
}
```
