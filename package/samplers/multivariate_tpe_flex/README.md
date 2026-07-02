---
author: Shuhei Watanabe
title: Multivariate TPE Sampler that considers dynamic value ranges
description: Multivariate TPESampler that includes past observations even after search space changes.
tags: [sampler, tpe, dynamic]
optuna_versions: [4.9.0]
license: MIT License
---

## Abstract

The original motivation can be found [here](https://github.com/optuna/optuna/issues/6299).

HPO is often iterative: a range that looked reasonable at the start can turn out to be too narrow once a few trials complete. For example, if `reg_alpha` for an XGBoost model is first searched over `[0, 1]` and the best trials keep landing near `0.99`, the natural next step is to widen it to, say, `[0, 2]` and keep optimizing on the same study.

Optuna's built-in `TPESampler(multivariate=True)` (and other samplers with the same requirement, e.g. `CmaEsSampler`) cannot make use of the trials collected before such a change. Internally, the relative search space is computed as the *intersection* of the distributions seen so far, so a parameter whose bounds ever changed is dropped from that intersection entirely; it then falls back to independent (univariate) sampling for that parameter, losing the joint/multivariate model between it and the rest of the parameters, even though every previous trial is still a perfectly valid observation once the range is only ever grown, never narrowed.

`TPESampler` in this package special-cases this scenario:

- `infer_relative_search_space` always returns the **most recently completed trial's** distributions as the current search space, instead of computing an intersection across all trials.
- Before fitting the below/above Parzen estimators, every historical trial is checked against this current search space: a trial is kept if each of its `FloatDistribution`/`IntDistribution` parameter values still falls inside the current `[low, high]` bounds (`CategoricalDistribution` choices are assumed immutable within a study, matching Optuna's own assumption, so those trials are always kept).

As long as a numerical parameter's range is only ever grown (the new bounds are a superset of every previous range), every past observation for it remains inside the new bounds and keeps contributing to the joint multivariate model. If a range is instead narrowed, trials whose recorded value now falls outside the new bounds are simply excluded from that round's model rather than raising an error.

This sampler is a fork of Optuna 4.9.0's `optuna.samplers.TPESampler` (kept under `_tpe_v4_9_0/`) with the deprecated, experimental `multivariate`/`group` constructor arguments removed: `multivariate` is fixed to `True` and group-decomposed conditional search spaces (`group=True`) are not supported.

## APIs

- `TPESampler(*, n_startup_trials: int = 10, n_ei_candidates: int = 24, seed: int | None = None, constant_liar: bool = True, constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None)`
  - `n_startup_trials`: The random sampling is used instead of the TPE algorithm until the given number of trials finish in the same study.
  - `n_ei_candidates`: Number of candidate samples used to calculate the expected improvement.
  - `seed`: Seed for random number generator.
  - `constant_liar`: If `True`, penalize running trials to avoid suggesting parameter configurations nearby. Defaults to `True` here, unlike Optuna's built-in `TPESampler`, which defaults to `False`.
  - `constraints_func`: An optional function that computes the objective constraints. It must take a `optuna.trial.FrozenTrial` and return the constraints. The return value must be a sequence of `float`s. A value strictly larger than 0 means that a constraint is violated. A value equal to or smaller than 0 is considered feasible. If `constraints_func` returns more than one value for a trial, that trial is considered feasible if and only if all values are equal to 0 or smaller. The function is evaluated after each successful trial and is not called when trials fail or are pruned.

## Installation

```shell
$ pip install -r https://hub.optuna.org/samplers/multivariate_tpe_flex/requirements.txt
```

## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


module = optunahub.load_module(package="samplers/multivariate_tpe_flex")
sampler = module.TPESampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)
```

### Growing the search space across `optimize` calls

The example below mirrors the motivating scenario: `reg_alpha` is first searched over `[0, 1]`, and once the sampler keeps favoring values near the upper bound, the same study is resumed with `reg_alpha` widened to `[0, 2]`. All trials from the first round remain valid observations for the second round because their `reg_alpha` values fall inside the new, wider bound.

```python
import optuna
import optunahub


def objective(trial: optuna.Trial, reg_alpha_high: float) -> float:
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, reg_alpha_high)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    return (reg_alpha - 0.95) ** 2 + (lr - 1e-2) ** 2


module = optunahub.load_module(package="samplers/multivariate_tpe_flex")
sampler = module.TPESampler(seed=0)
study = optuna.create_study(sampler=sampler)

# Round 1: reg_alpha is initially assumed to live in [0, 1].
study.optimize(lambda trial: objective(trial, reg_alpha_high=1.0), n_trials=30)

# Round 2: reg_alpha kept landing near 1.0, so the range is widened to [0, 2] and
# optimization continues on the same study/sampler, still using the round 1 trials.
study.optimize(lambda trial: objective(trial, reg_alpha_high=2.0), n_trials=30)

print(study.best_params, study.best_value)
```
