---
author: Hiroaki Natsume
title: NSGAII sampler with Initial Trials
description: Sampler using NSGAII algorithm with initial trials.
tags: [Sampler, Multi-Objective, Genetic Algorithm]
optuna_versions: [4.0.0]
license: MIT License
---

## Abstract

If Optuna's built-in NSGAII has a study obtained from another sampler, but continues with that study, it cannot be used as the first generation, and optimization starts from zero.
This means that even if you already know good individuals, you cannot use it in the GA.
In this implementation, the already sampled results are included in the initial individuals of the GA to perform the optimization.

Note, however, that this has the effect that the implementation does not necessarily support multi-threading in the generation of the initial generation.
After the initial generation, the implementation is similar to the built-in NSGAII.

## Class or Function Names

- NSGAIIwITSampler

## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)
    v0 = 4 * x**2 + 4 * y**2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


storage = optuna.storages.InMemoryStorage()

# Sampling 0 generation using enqueueing & qmc sampler
study = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=optuna.samplers.QMCSampler(seed=42),
    study_name="test",
    storage=storage,
)
study.enqueue_trial(
    {
        "x": 0,
        "y": 0,
    }
)
study.optimize(objective, n_trials=128)

# Using sampling results as the initial generation
sampler = optunahub.load_module(
    "samplers/nsgaii_with_initial_trials",
).NSGAIIwITSampler(population_size=25, seed=42)

study = optuna.create_study(
    directions=["minimize", "minimize"],
    sampler=sampler,
    study_name="test",
    storage=storage,
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)

optuna.visualization.plot_pareto_front(study).show()
```

## Others

The implementation is similar to Optuna's NSGAII except for the handling of initial generations. The license and documentation are below.

- [Documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html)
- [License](https://github.com/optuna/optuna/blob/master/LICENSE)
