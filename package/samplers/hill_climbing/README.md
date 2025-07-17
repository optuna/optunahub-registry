---
author: Ayush Chaudhary <Ayushkumar.chaudhary2003@gmail.com>
title: Hill Climbing Sampler
description: Hill climbing algorithm for discrete optimization problems
tags: [sampler, hill-climbing, discrete-optimization, local-search]
optuna_versions: [4.0.0]
license: MIT
---

## Abstract

The hill climbing algorithm is an optimization technique that iteratively improves a solution by evaluating neighboring solutions in search of a local maximum or minimum. Starting with an initial guess, the algorithm examines nearby "neighbor" solutions, moving to a better neighbor if one is found. This process continues until no improvement can be made locally, at which point the algorithm may restart from a new random position.

This implementation focuses on discrete optimization problems, supporting integer and categorical parameters only. The sampler fully supports both minimization and maximization objectives as specified in the Optuna study direction, making it compatible with all standard Optuna optimization workflows.

## Class or Function Names

- **HillClimbingSampler**

## Installation

No additional dependencies are required beyond Optuna and OptunaHub.

```bash
pip install optuna optunahub
```

## APIs

### HillClimbingSampler

```python
HillClimbingSampler(
    search_space: dict[str, BaseDistribution] | None = None,
    *,
    seed: int | None = None,
    neighbor_size: int = 5,
    max_restarts: int = 10,
)
```

#### Parameters

- **search_space** (dict\[str, BaseDistribution\] | None, optional): A dictionary containing the parameter names and their distributions. If None, the search space is inferred from the study.
- **seed** (int | None, optional): Seed for the random number generator to ensure reproducible results.
- **neighbor_size** (int, default=5): Number of neighboring solutions to generate and evaluate in each iteration.
- **max_restarts** (int, default=10): Maximum number of times the algorithm will restart from a random position when no improvements are found.

## Supported Distributions

- **IntDistribution**: Integer parameters with specified bounds
- **CategoricalDistribution**: Categorical parameters with discrete choices

## Supported Study Directions

- **Minimization**: `optuna.create_study(direction="minimize")` or `optuna.create_study(direction=optuna.study.StudyDirection.MINIMIZE)`
- **Maximization**: `optuna.create_study(direction="maximize")` or `optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)`

The algorithm automatically adapts its improvement criteria and best solution tracking based on the study direction, ensuring optimal performance for both optimization types.

## Limitations

- **Discrete only**: This sampler only supports discrete parameter types (`suggest_int` and `suggest_categorical`). Continuous parameters (`suggest_float`) are not supported.
- **Single-objective**: Only single-objective optimization is supported.

## Examples

### Basic Usage with Minimization

````python
import optuna
import optunahub

```python
import optuna
import optunahub

def objective(trial):
    # Integer parameter
    x = trial.suggest_int("x", -10, 10)
    # Categorical parameter  
    algorithm = trial.suggest_categorical("algorithm", ["A", "B", "C"])
    # Simple objective function
    penalty = {"A": 0, "B": 1, "C": 2}[algorithm]
    return x**2 + penalty

# Load the hill climbing sampler
module = optunahub.load_module("samplers/hill_climbing")
sampler = module.HillClimbingSampler(
    neighbor_size=8,    # Generate 8 neighbors per iteration
    max_restarts=5,     # Allow up to 5 restarts
    seed=42             # For reproducible results
)

# Create study for minimization
study = optuna.create_study(sampler=sampler, direction="minimize")
study.optimize(objective, n_trials=100)

print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")
````
