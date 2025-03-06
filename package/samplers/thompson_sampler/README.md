---
author: Samuel D. McDermott
title: Thompson Sampler
description: Sampler based on Thompson sampling for categorical variables.
tags: [sampler, Thompson sampling, categorical variables]
optuna_versions: [4.2.1]
license: MIT License
---

## Class or Function Names

- ThompsonSampler

## Example

```python
import optunahub
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

mod = optunahub.load_module("samplers/thompson_sampler")
sampler = mod.ThompsonSampler()

def gaussians(x: float, label: str):
    """
    Args:
        x: float: nuisance parameter
        label: str: determines which of four Gaussians we sample from
    
    Returns:
        a Gaussian sample from one of four different normal distributions, depending on the choice of `label` 
    """
    if label == 'a':
        return np.random.normal(loc = 1, scale = 8)
    elif label == 'b':
        return np.random.normal(loc = 5, scale = 2)
    elif label == 'c':
        return np.random.normal(loc = 0, scale = 3)
    else:
        return np.random.normal(loc = 2, scale = 2)

def objective(trial):
    xv = trial.suggest_float('x', -1, 1)
    label = trial.suggest_categorical('label', ['a', 'b', 'c', 'd'])
    return gaussians(xv, label)


package_name = "package/samplers/thompson_sampler"
sampler = optunahub.load_module(
    package=package_name
).ThompsonSampler()

study_T = optuna.create_study(direction='maximize',
                            sampler=sampler)
study_T.optimize(objective, n_trials=111)

study_base = optuna.create_study(direction='maximize')
study_base.optimize(objective, n_trials=111)
lab_dict = defaultdict(list)
for i in study_base.trials:
    lab_dict[i.params['label']].append(i.values[0])
    
for k, v in study_T.categorical_variable_samples.items():
    print(f"label {k}:")
    print(f"\tThompson sampler: max = {max(v)} from {len(v)} samples")
    print(f"\tBase sampler: max = {max(lab_dict[k])} from {len(lab_dict[k])} samples")
    print("\n")
```

The base sampler follows a "winner takes all" approach, whereas Thompson sampling does a better job of balancing exploration and exploitation.

## Others

This package provides a sampler based on the principles of Thompson sampling. For a pedagogical introduction, see [A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038).
