---
author: Hiroaki Natsume
title: HypE Sampler
description: Sampler using HypE (Hypervolume Estimation Algorithm) for many-objective optimization. It uses Quasi-Monte Carlo sampling for efficient hypervolume estimation.
tags: [Sampler, Multi-Objective, Many-Objective, GA, Warmstart]
optuna_versions: [4.5.0]
license: MIT License
---

## Abstract

HypE (Hypervolume Estimation Algorithm) is a fast hypervolume-based evolutionary algorithm designed for many-objective optimization problems.

Unlike traditional hypervolume-based methods that become computationally expensive with increasing objectives, HypE uses Monte Carlo sampling to efficiently estimate hypervolume contributions.

It employs a greedy selection strategy that preferentially retains individuals with higher hypervolume contributions, enabling effective convergence toward the Pareto front.

## APIs

- `HypESampler(*, population_size=50, n_samples=1024, mutation=None, mutation_prob=None, crossover=None, crossover_prob=0.9, seed=None)`
  - `population_size`: Size of the population for the evolutionary algorithm. Default is 50.
  - `n_samples`: Number of samples for hypervolume estimation using Quasi-Monte Carlo. Should be a power of 2 for optimal Sobol sequence performance. Default is 1024.
  - `mutation`: Mutation to be applied when creating child individuals. If `None`, `UniformMutation` is selected.
    - For categorical variables, it is always `UniformMutation`.
  - `mutation_prob`: Probability of mutation for each parameter. If `None`, it defaults to `1.0 / n_params`.
  - `crossover`: Crossover to be applied when creating child individuals. If `None`, `UniformCrossover` is selected.
  - `crossover_prob`: Probability of crossover. Default is 0.9.
  - `seed`: Random seed for reproducibility.

### Supported Mutation Methods

- `UniformMutation()`
  - Mutation using a uniform distribution for generating new values.
- `PolynomialMutation(eta=20)`
  - Mutation using a polynomial distribution.
  - `eta`: Distribution index. Larger values produce narrower distributions. Recommended range: `eta in [20, 100]`.
- `GaussianMutation(sigma_factor=1/30)`
  - Mutation using a Gaussian distribution.
  - `sigma_factor`: Factor multiplied by the search range to determine sigma. When `sigma_factor=1.0`, sigma equals the full search range.

## Example

Here is the code example comparing hypervolumes using the WFG1 function with NSGA-III.

```python
import optuna
import optunahub

n_objs = 5
seed = 42
population_size = 50
n_trials = 1000

wfg = optunahub.load_module("benchmarks/wfg")
wfg1 = wfg.Problem(function_id=1, n_objectives=n_objs, dimension=10)

mod = optunahub.load_module("samplers/hype")
mutation = mod.PolynomialMutation()
crossover = optuna.samplers.nsgaii.SBXCrossover()

samplers = [
    mod.HypESampler(
        population_size=population_size,
        mutation=mutation,
        crossover=crossover,
        seed=seed,
    ),
    optuna.samplers.NSGAIIISampler(
        population_size=population_size, crossover=crossover, seed=seed
    ),
]
studies = []
for sampler in samplers:
    study = optuna.create_study(
        sampler=sampler,
        study_name=f"{sampler.__class__.__name__}",
        directions=["minimize"] * n_objs,
    )
    study.optimize(wfg1, n_trials=n_trials)
    studies.append(study)

reference_point = [3 * (i + 1) for i in range(n_objs)]
m = optunahub.load_module("visualization/plot_hypervolume_history_multi")
fig = m.plot_hypervolume_history(studies, reference_point)
fig.show()
```

## Reference

Johannes Bader, Eckart Zitzler; HypE: An Algorithm for Fast Hypervolume-Based Many-Objective Optimization. Evol Comput 2011; 19 (1): 45–76. [DOI](https://doi.org/10.1162/EVCO_a_00009)

### Difference from the Original Paper

This implementation differs from the original HypE paper in the sampling method:

| Aspect           | Original Paper              | This Implementation                |
| ---------------- | --------------------------- | ---------------------------------- |
| Sampling Method  | Monte Carlo (pseudo-random) | Quasi-Monte Carlo (Sobol sequence) |
| Convergence Rate | O(1/√N)                     | O(1/N)                             |
| Space Coverage   | Random                      | Low-discrepancy, more uniform      |

The use of QMC allows achieving equivalent estimation accuracy with fewer samples, resulting in faster execution while maintaining solution quality. For optimal QMC performance, `n_samples` should be a power of 2 (e.g., 1024, 2048, 4096).
