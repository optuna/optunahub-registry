---
author: Hiroaki Natsume
title: SPEAII sampler
description: Sampler using SPEA-II algorithm, a multi-objective evolutionary algorithm that maintains an external archive of non-dominated solutions. It supports custom mutation methods and warm-start optimization.
tags: [Sampler, Multi-Objective, Warmstart, GA]
optuna_versions: [4.5.0]
license: MIT License
---

## Abstract

SPEA-II (Strength Pareto Evolutionary Algorithm 2) is an improved multi-objective evolutionary algorithm that differs from NSGA-II in its selection mechanism. While NSGA-II uses non-dominated sorting and crowding distance, SPEA-II maintains an external archive to preserve elite non-dominated solutions and uses a fine-grained fitness assignment strategy based on the strength of domination.

Note that when using warm-start with existing trials, the initial generation may not support concurrent sampling. After the initial generation, the implementation follows standard evolutionary algorithm parallelization.

## APIs

- `SPEAIISampler(*, population_size=50, archive_size=None, mutation=None, mutation_prob=None, crossover=None, crossover_prob=0.9, seed=None)`
  - `archive_size`: Size of the external archive that stores elite non-dominated solutions. The archive is used in the SPEA-II selection process to maintain diversity and quality of the Pareto front. If `None`, it defaults to `population_size`.
  - `mutation`: Mutation to be applied when creating child individual. If `None`, `UniformMutation` is selected.
    - For categorical variables, it is always `UniformMutation`.
  - The other arguments are the same as for Optuna's NSGA-II.
  - Supported mutation methods are listed below:
    - `UniformMutation()`
      - This is a mutation method that uses a Uniform distribution for the distribution of the generated individuals.
    - `PolynomialMutation(eta=20)`
      - This is a mutation method that uses a Polynomial distribution for the distribution of the generated individuals.
      - `eta`: Argument for the width of the distribution. The larger the value, the narrower the distribution. A value `eta ∈ [20, 100]` is adequate in most problems.
    - `GaussianMutation(sigma_factor=1/30)`
      - This is a mutation method that uses a Gaussian distribution for the distribution of the generated individuals.
      - `sigma_factor`: It is a factor that is multiplied by the sigma of the Gaussian distribution. When the `sigma_factor` is `1.0`, the sigma is the difference between the maximum and minimum of the search range for the target variable.

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


module = optunahub.load_module(
    "samplers/speaii",
)
mutation = module.PolynomialMutation(eta=20)
sampler = module.SPEAIISampler(population_size=50, archive_size=50, mutation=mutation)

study = optuna.create_study(
    sampler=sampler,
    directions=["minimize", "minimize"],
)
study.optimize(objective, n_trials=1000)

optuna.visualization.plot_pareto_front(study).show()
```

### Reference

- SPEA-II
  - Zitzler, Eckart., Laumanns, Marco., & Thiele, Lothar. (2001). SPEA2: Improving the Strength Pareto Evolutionary Algorithm. TIK-Report, 103.
- Mutation
  - Deb, Kalyanmoy., & Deb, Debayan. (2014). Analysing Mutation Schemes for Real-parameter Genetic Algorithms. International Journal of Artificial Intelligence and Soft Computing, 4(1), 1-28.
