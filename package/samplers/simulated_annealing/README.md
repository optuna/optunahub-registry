---
author: 'Optuna team'
title: 'Simulated Annealing Sampler'
description: 'Sampler based on simulated annealing algorithm.'
tags: ['sampler', 'simulated annealing']
optuna_versions: [3.5, 3.6]
license: 'MIT'
---

Simulated Annealing Sampler
===

This package provides a sampler based on Simulated Annealing algorithm.
For more details, see [the documentation](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/005_user_defined_sampler.html).


# SimulatedAnnealingSampler

```python
class SimulatedAnnealingSampler(temperature=100)
```

Sampler based on Simulated Annealing algorithm.

## Parameters
- `temperature=100 (int)` - A temperature parameter for simulated annealing.


## Example

```python
import optunahub
import optuna


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 0, 1)

    return x


if __name__ == "__main__":
    mod = optunahub.load("sampler/simulated_annealing")

    sampler = mod.SimulatedAnnealingSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=20)

    print(study.best_trial.value, study.best_trial.params)
```

# Author Information

This package is contributed by [Optuna team](https://github.com/orgs/optuna/people).
