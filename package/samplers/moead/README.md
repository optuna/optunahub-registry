---
author: Hiroaki Natsume
title: MOEA/D sampler
description: Sampler using MOEA/D algorithm. MOEA/D stands for "Multi-Objective Evolutionary Algorithm based on Decomposition.
tags: [Sampler, Multi-Objective Optimization, Evolutionary Algorithms]
optuna_versions: [4.0.0]
license: MIT License
---

## Abstract

Sampler using MOEA/D algorithm. MOEA/D stands for "Multi-Objective Evolutionary Algorithm based on Decomposition.

This sampler is specialized for multiobjective optimization. The objective function is internally decomposed into multiple single-objective subproblems to perform optimization.

It may not work well with multi-threading. Check results carefully.

## APIs

- `MOEADSampler(*, population_size = 100, n_neighbors = None, scalar_aggregation_func = "tchebycheff", mutation = None, mutation_prob = None, crossover = None, crossover_prob = 0.9, seed = None`
  - `n_neighbors`: The number of the weight vectors in the neighborhood of each weight vector. The larger this value, the more weight is applied to the exploration.
    - If None, `population_size // 10` is used
  - `scalar_aggregation_func`: The scalar aggregation function to use. The default is `tchebycheff`. Other options is `weight_sum`.
  - `mutation`: Mutation to be applied when creating child individual.
    - If None, `UniformMutation` is selected. For categorical variables, it is always `UniformMutation`.
  - `crossover`: Crossover to be applied when creating child individual.
    - If None, `UniformCrossover(swapping_prob=0.5)` is selected.
- The other arguments are the same as for Optuna's NSGA-II.
- Supported mutation methods are listed below
  - `UniformMutation()`
    - This is a mutation method that uses a Uniform distribution for the distribution of the generated individuals.
  - `PolynomialMutation(eta=20)`
    - This is a mutation method that uses a Polynomial distribution for the distribution of the generated individuals.
    - `eta`: Argument for the width of the distribution. The larger the value, the narrower the distribution. A value `eta ∈ [20, 100]` is adequate in most problems
  - `GaussianMutation(sigma_factor=1/30)`
    - This is a mutation method that uses a Gaussian distribution for the distribution of the generated individuals.
    - `sigma_factor`: It is a factor that is multiplied by the sigma of the Gaussian distribution. When the `sigma_factor` is `1.0`, the sigma is the difference between the maximum and minimum of the search range for the target variable.

## Installation

```
pip install scipy
```

or

```
pip install -r https://hub.optuna.org/samplers/moead/requirements.txt
```

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


mod = optunahub.load_module("samplers/moead")
sampler = mod.MOEADSampler(
    population_size=100,
    scalar_aggregation_func="tchebycheff",
    n_neighbors=20,
    mutation=mod.PolynomialMutation(eta=20)
)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=1000)
```

## Others

For more information, check out [Optuna's Medium article](https://medium.com/optuna/an-introduction-to-moea-d-and-examples-of-multi-objective-optimization-comparisons-8630565a4e89)

### Reference

Q. Zhang and H. Li,
"MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, Dec. 2007,
[doi: 10.1109/TEVC.2007.892759](https://doi.org/10.1109/TEVC.2007.892759).
