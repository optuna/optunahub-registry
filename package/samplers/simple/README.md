---
author: 'Optuna team'
title: 'Simple Sampler'
description: 'An easy sampler base class to implement custom samplers.'
tags: ['sampler', 'development']
optuna_versions: [3.6.1]
license: 'MIT'
---

Simple Sampler
===

This package provides an easy sampler base class to implement custom samplers.
You can make your own sampler easily by inheriting `SimpleSampler` and by implementing necessary methods.


# SimpleSampler

```python
class SimpleSampler(search_space)
```


## Parameters
- `search_space (dict[str, BaseDistribution])` - A search space for the objective function.


## Example

```python
from typing import Any

import numpy as np
import optuna
import optunahub
from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.trial import FrozenTrial


class UserDefinedSampler(
    optunahub.load_module("samplers/simple").SimpleSampler
):
    def __init__(self, search_space: dict[str, BaseDistribution]) -> None:
        super().__init__(search_space)
        self._rng = np.random.RandomState()

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        params = {}
        for n, d in search_space.items():
            if isinstance(d, FloatDistribution):
                params[n] = self._rng.uniform(d.low, d.high)
            elif isinstance(d, IntDistribution):
                params[n] = self._rng.randint(d.low, d.high)
            else:
                raise ValueError("Unsupported distribution")
        return params


if __name__ == "__main__":

    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0, 1)

        return x

    sampler = UserDefinedSampler({"x": FloatDistribution(0, 1)})
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)

    print(study.best_trial.value, study.best_trial.params)
```

# Author Information

This package is contributed by [Optuna team](https://github.com/orgs/optuna/people).
