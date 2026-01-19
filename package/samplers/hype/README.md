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

- `HypESampler(*, population_size=50, n_samples=4096, mutation=None, mutation_prob=None, crossover=None, crossover_prob=0.9, hypervolume_method="auto", seed=None)`
  - `population_size`: Size of the population for the evolutionary algorithm. Default is 50.
  - `n_samples`: Number of samples for hypervolume estimation using Quasi-Monte Carlo. Should be a power of 2 for optimal Sobol sequence performance. Default is 4096.
  - `hypervolume_method`: Method for hypervolume contribution calculation. If "auto", "exact" is used when the number of objectives is 3 or less, and "estimation" is used otherwise, following the original HypE paper. If "exact", exact hypervolume calculation is always used. If "estimation", Monte Carlo estimation is always used. Default is "auto".
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

## Installation

```
pip install scipy
```

or

```
pip install -r https://hub.optuna.org/samplers/hype/requirements.txt
```

## Example

Here is the code example comparing hypervolumes using the WFG1 function with NSGA-III.

```python
import optuna
import optunahub

n_objs = 4
seed = 42
population_size = 50
n_trials = 1000

wfg = optunahub.load_module("benchmarks/wfg")
wfg1 = wfg.Problem(function_id=1, n_objectives=n_objs, dimension=10)

mod = optunahub.load_module("samplers/hype")
mutation = mod.PolynomialMutation()
crossover = optuna.samplers.nsgaii.SBXCrossover()

sampler = mod.HypESampler(
  population_size=population_size,
  n_samples=4096,
  hypervolume_method="auto",
  mutation=mutation,
  crossover=crossover,
  seed=seed,
)
study = optuna.create_study(
    sampler=sampler,
    study_name=f"{sampler.__class__.__name__}",
    directions=["minimize"] * n_objs,
)
study.optimize(wfg1, n_trials=n_trials)
```

## Reference

Johannes Bader, Eckart Zitzler; HypE: An Algorithm for Fast Hypervolume-Based Many-Objective Optimization. Evol Comput 2011; 19 (1): 45–76. [DOI](https://doi.org/10.1162/EVCO_a_00009)

## Differences from the Original Paper

### 1. Quasi-Monte Carlo Sampling

This implementation uses Quasi-Monte Carlo (QMC) with Sobol sequences instead of standard Monte Carlo sampling:

| Aspect           | Original Paper              | This Implementation                |
| ---------------- | --------------------------- | ---------------------------------- |
| Sampling Method  | Monte Carlo (pseudo-random) | Quasi-Monte Carlo (Sobol sequence) |
| Convergence Rate | O(1/√N)                     | O(1/N)                             |
| Space Coverage   | Random                      | Low-discrepancy, more uniform      |

The use of QMC allows achieving equivalent estimation accuracy with fewer samples, resulting in faster execution while maintaining solution quality. For optimal QMC performance, `n_samples` should be a power of 2 (e.g., 1024, 2048, 4096).

The original paper recommends 10000 samples for Monte Carlo estimation. However, since QMC has a faster convergence rate (O(1/N) vs O(1/√N)), fewer samples are needed to achieve equivalent accuracy:

| Samples | Monte Carlo Relative Error | QMC (Sobol) Relative Error |
| ------- | -------------------------- | -------------------------- |
| 10000   | 1/√10000 = 0.01            | 1/10000 = 0.0001           |
| 4096    | 1/√4096 ≈ 0.0156           | 1/4096 ≈ 0.000244          |
| 1024    | 1/√1024 ≈ 0.0313           | 1/1024 ≈ 0.000977          |

The default of 4096 provides a conservative balance between accuracy and computational cost.

Note: The table above shows theoretical relative comparisons between sampling methods. The optimal number of samples depends on factors such as the number of objectives, the smoothness of the objective functions, and the shape of the Pareto front. For higher accuracy requirements, consider increasing `n_samples` (e.g., 8192 or 16384).

### 2. Configurable Hypervolume Calculation Method

The original paper automatically switches between exact hypervolume calculation (for 3 or fewer objectives) and Monte Carlo estimation (for 4 or more objectives). This implementation provides the `hypervolume_method` parameter to explicitly control this behavior:

| Value        | Behavior                                                                |
| ------------ | ----------------------------------------------------------------------- |
| "auto"       | Use exact calculation for ≤3 objectives, estimation otherwise (default) |
| "exact"      | Always use exact hypervolume calculation                                |
| "estimation" | Always use QMC-based estimation                                         |

This flexibility allows users to:

- Force exact calculation for higher accuracy when computational cost is acceptable
- Force estimation for faster execution with many low-dimensional problems
- Use the paper's recommended automatic switching (default)
