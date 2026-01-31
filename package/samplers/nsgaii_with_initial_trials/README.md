---
author: Hiroaki Natsume
title: NSGAII sampler with Initial Trials
description: Sampler using NSGAII algorithm with initial trials. It also supports the selection of mutation methods.
tags: [Sampler, Multi-Objective, GA, Warmstart]
optuna_versions: [4.5.0]
license: MIT License
---

## Abstract

If Optuna's built-in NSGAII has a study obtained from another sampler, but continues with that study, it cannot be used as the first generation, and optimization starts from zero.
This means that even if you already know good individuals, you cannot use it in the GA.

In this implementation, the already sampled results are included in the initial individuals of the GA to perform the optimization.

Note, however, that this has the effect that the implementation does not necessarily support multi-threading in the generation of the initial generation.
After the initial generation, the implementation is similar to the built-in NSGAII.

### Differences from Optuna's NSGA-II

- If an existing trial exists, it will be used as the initial generation
- Multiple mutation methods are available for selection
- During tournament selection for crossover calculations, congestion distance is considered
  - Optuna's NSGA-II performs parent tournament selection based solely on dominance relationships without considering congestion distance

## APIs

- `NSGAIIwITSampler(*, mutation=None, population_size=50, mutation_prob=None, crossover=None, crossover_prob=0.9, swapping_prob=0.5, seed=None, constraints_func=None, elite_population_selection_strategy=None, after_trial_strategy=None)`
  - `mutation`: Mutation to be applied when creating child individual. If None, `UniformMutation` is selected.
    - For categorical variables, it is always `UniformMutation`.
  - The other arguments are the same as for Optuna's NSGA-II.

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
seed = 42

# Sampling 0 generation using enqueueing & qmc sampler
study = optuna.create_study(
    directions=directions,
    sampler=optuna.samplers.QMCSampler(seed=seed),
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
sampler = module.NSGAIIwITSampler(population_size=25, seed=seed, mutation=mutation)

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

The implementation is modified Optuna's NSGAII to consider initial trials and mutation. Its license and documentation are below.

- [Documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIISampler.html)
- [License](https://github.com/optuna/optuna/blob/master/LICENSE)

### Reference

- Mutation
  - Deb, Kalyanmoy., & Deb, Debayan. (2014). Analysing Mutation Schemes for Real-parameter Genetic Algorithms. International Journal of Artificial Intelligence and Soft Computing, 4(1), 1-28.
