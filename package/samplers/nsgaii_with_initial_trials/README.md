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

In addition, enhancements to Optuna's NSGA-II include the option to select mutation methods.

## APIs

- `NSGAIIwITSampler(*, mutation=None, population_size=50, mutation_prob=None, crossover=None, crossover_prob=0.9, swapping_prob=0.5, seed=None, constraints_func=None, elite_population_selection_strategy=None, after_trial_strategy=None)`
  - `mutation`: Mutation to be applied when creating child individual. If None, `UniformMutation` is selected.
    - [Kalyanmoy Deb and Debayan Deb. 2014. Analysing mutation schemes for real-parameter genetic algorithms. Int. J. Artif. Intell. Soft Comput. 4, 1 (February 2014), 1–28.](https://doi.org/10.1504/IJAISC.2014.059280)
  - For categorical variables, it is always `UniformMutation`.
  - Supported mutation methods are listed below
    - `UniformMutation()`
      - This is a mutation method that uses a Uniform distribution for the distribution of the generated individuals.
    - `PolynomialMutation(eta=20)`
      - This is a mutation method that uses a Polynomial distribution for the distribution of the generated individuals.
      - `eta`: Argument for the width of the distribution. The larger the value, the narrower the distribution. A value `eta ∈ [20, 100]` is adequate in most problems
    - `GaussianMutation(sigma_factor=1/30)`
      - This is a mutation method that uses a Gaussian distribution for the distribution of the generated individuals.
      - `sigma_factor`: It is a factor that is multiplied by the sigma of the Gaussian distribution. When the `sigma_factor` is `1.0`, the sigma is the difference between the maximum and minimum of the search range for the target variable.
  - The other arguments are the same as for Optuna's NSGA-II.

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
study_name = "test"
directions = ["minimize", "minimize"]

# Sampling 0 generation using enqueueing & qmc sampler
study = optuna.create_study(
    directions=directions,
    sampler=optuna.samplers.QMCSampler(seed=42),
    study_name=study_name,
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
module = optunahub.load_module(
    "samplers/nsgaii_with_initial_trials",
)
mutation = module.PolynomialMutation(eta=20)
sampler = module.NSGAIIwITSampler(population_size=25, seed=42, mutation=mutation)

study = optuna.create_study(
    directions=directions,
    sampler=sampler,
    study_name=study_name,
    storage=storage,
    load_if_exists=True,
)
study.optimize(objective, n_trials=100)

optuna.visualization.plot_pareto_front(study).show()
```

## Others

The implementation is similar to Optuna's NSGAII except for the handling of initial generations and mutation. The license and documentation are below.

- [Documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html)
- [License](https://github.com/optuna/optuna/blob/master/LICENSE)
