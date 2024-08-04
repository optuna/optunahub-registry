---
author: Shuhei Watanabe
title: Tree-Structured Parzen Estimator; Understanding Its Algorithm Components and Their Roles for Better Empirical Performance
description: The optimizer that reproduces the algorithm described in the paper ``Tree-Structured Parzen Estimator; Understanding Its Algorithm Components and Their Roles for Better Empirical Performance''.
tags: [sampler, tpe, paper, research]
optuna_versions: [v4.0.0b0]
license: MIT License
---

## Class or Function Names

- CustomizableTPESampler

## Example

This sampler can be used in this way.

```python
import numpy as np

import optuna

import optunahub


def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", -5, 5)
    z = trial.suggest_categorical("z", ["a", "aa", "aaa"])
    return len(z) * (x**2 + y**2)


module = optunahub.load_module(package="samplers/tpe_tutorial", repo_owner="nabenabe0928", ref="add-tpe-tutorial")
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
arg_choices = {
    "consider_prior": [True, False],
    "consider_magic_clip": [True, False],
    "multivariate": [True, False],
    "b_magic_exponent": [0.5, 1.0, 2.0, np.inf],
    "min_bandwidth_factor": [0.01, 0.1],
    "gamma_strategy": ["linear", "sqrt"],
    "weight_strategy": ["uniform", "old-decay", "old-drop", "EI"],
    "bandwidth_strategy": ["optuna", "hyperopt", "scott"],
    "categorical_prior_weight": [0.1, None],
}
for arg_name, choices in arg_choices.items():
    results = []
    for choice in choices:
        print(arg_name, choice)
        sampler = module.CustomizableTPESampler(seed=0, **{arg_name: choice})
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=100 if arg_name != "b_magic_exponent" else 200)
        results.append(study.trials[-1].value)

    print(f"Did every setup yield different results for {arg_name}?: {len(set(results)) == len(results)}")

```

In the paper, the following arguments, which do not exist in Optuna, were researched:

- `gamma_strategy`: The splitting algorithm in Table 3. The choices are `linear` and `sqrt`.
- `gamma_beta`: The beta parameter for the splitting algorithm in Table 3. This value must be positive.
- `weight_strategy`: The weighting algorithm in Table 3. The choices are `uniform`, `old-decay`, `old-drop`, and `EI`.
- `categorical_prior_weight`: The categorical bandwidth in Table 3. If `None`, the Optuna default algorithm will be used.
- `bandwidth_strategy`: The bandwidth selection heuristic in Table 6. The choices are `optuna`, `hyperopt`, and `scott`.
- `min_bandwidth_factor`: The minimum bandwidth factor in Table 6. This value must be positive.
- `b_magic_exponent`: The exponent alpha in Table 6. Optuna takes 1.0 by default.

For more details, please check the paper.

### Bibtex

When you use this sampler, please cite the following:

```bibtex
@inproceedings{watanabe_tpe_tutorial2023,
    title={Tree-Structured {P}arzen Estimator: Understanding Its Algorithm Components and Their Roles for Better Empirical Performance},
    author={Shuhei Watanabe},
    booktitle={arXiv:2304.11127},
    year={2023}
}
```
