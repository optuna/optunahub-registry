---
author: Luca Bernstiel
title: Particle Swarm Optimization (PSO) Sampler
description: Particle Swarm Optimization is a population-based stochastic optimization algorithm inspired by flocking behavior, where particles iteratively adjust their positions using personal and global bests to search for optima.
tags: [sampler]
optuna_versions: [4.5.0]
license: MIT License
---

## Abstract

Particle Swarm Optimization (PSO) is a population-based stochastic optimizer inspired by flocking behavior, where particles iteratively adjust their positions using personal and global bests to search for optima. This sampler supports single-objective, continuous optimization only.

> Note: Categorical distributions are suggested by the underlaying RandomSampler.

> Note: Multi-objective optimization is not supported.

For details on the algorithm, see Kennedy and Eberhart (1995): [Particle Swarm Optimization](https://doi.org/10.1109/ICNN.1995.488968).

## APIs

- `PSOSampler(search_space: dict[str, BaseDistribution] | None = None, n_particles: int = 10, inertia: float = 0.5, cognitive: float = 1.5, social: float = 1.5, seed: int | None = None)`
  - `search_space`: A dictionary containing the search space that defines the parameter space. The keys are the parameter names and the values are [the parameter's distribution](https://optuna.readthedocs.io/en/stable/reference/distributions.html). If the search space is not provided, the sampler will infer the search space dynamically.
    Example:
    ```python
    search_space = {
        "x": optuna.distributions.FloatDistribution(-510, 10),
        "y": optuna.distributions.FloatDistribution(-10, 10),
    }
    PSOSampler(search_space=search_space)
    ```
  - `n_particles`: Number of particles (population size). Prefer total n_trials to be a multiple of n_particles to run full PSO iterations. Larger swarms can improve exploration when budget allows.
  - `inertia`: Inertia weight controlling momentum (influence of previous velocity). Higher values favor exploration, lower favor exploitation.
  - `cognitive`: Personal-best acceleration coefficient (c1). Controls attraction toward each particle’s own best.
  - `social`: Global-best acceleration coefficient (c2). Controls attraction toward the swarm’s best.
  - `seed`: Seed for random number generator.

## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return x**2 + y**2

n_trials = 100
n_generations = 5

sampler = optunahub.load_module(package="samplers/pso").PSOSampler(
    {
        "x": optuna.distributions.FloatDistribution(-10, 10),
        "y": optuna.distributions.FloatDistribution(-10, 10, step=0.1),
    },
    n_particles=int(n_trials / n_generations),
    inertia=0.5,
    cognitive=1.5,
    social=1.5,
)

study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=n_trials)
print(study.best_trials)
```
