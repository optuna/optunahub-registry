---
author: Optuna Team
title: Multi-dimensional Knapsack Problem
description: The multi-dimensional knapsack problem is a combinatorial optimization problem that generalizes the classic knapsack problem to multiple dimensions.
tags: [benchmark, discrete optimization, combinatorial optimization, real world problem]
optuna_versions: [4.5.0]
license: MIT License
---

## Abstract

The Multi-dimensional Knapsack Problem (MKP) is a fundamental combinatorial optimization problem that generalizes the classic knapsack problem to multiple dimensions. In this problem, each item has multiple attributes (e.g., weight, volume, size) and the goal is to maximize the total value of selected items while satisfying constraints on each attribute. Despite its conceptual simplicity, the MKP is NP-hard and appears frequently in real-world applications, such as resource allocation, capital budgeting, and project selection, as remarked in recent surveys e.g.,  [Zamuda et al., 2018](https://doi.org/10.1145/3205651.3208307) and [Skackauskas and Kalganova, 2022](https://doi.org/10.1016/j.sasc.2022.200041).

The mathematical formulation is:

```
maximize:   sum_{i=1}^n v_i * x_i
subject to: sum_{i=1}^n w_ij * x_i <= c_j  for j = 1, ..., m
            x_i in {0, 1}  for i = 1, ..., n
```

where:

- n = number of items
- m = number of dimensions (constraints)
- v_i = value of item i
- w_ij = weight of item i in dimension j
- c_j = capacity constraint for dimension j
- x_i = binary variable indicating whether item i is selected

## APIs

### class `Problem(n_items: int = 20, n_dimensions: int = 2, seed: int | None = None, max_value: int = 100, max_weight: int = 50, max_capacity: float = 0.5)`

- `n_items`: Number of items in the problem instance (default: 20).
- `n_dimensions`: Number of dimensions/constraints (default: 2).
- `seed`: Random seed for generating problem instance. If None, uses current random state.
- `max_value`: Maximum value for randomly generated item values (default: 100).
- `max_weight`: Maximum weight for randomly generated item weights (default: 50).
- `max_capacity`: Capacity ratio relative to total weights (default: 0.5, meaning 50% of total weights).

#### Methods and Properties

- `search_space`: Return the search space.
  - Returns: `dict[str, optuna.distributions.BaseDistribution]`
- `directions`: Return the optimization directions (maximize).
  - Returns: `list[optuna.study.StudyDirection]`
- `evaluate(params: dict[str, int])`: Evaluate the objective function given a dictionary of parameters.
  - Args:
    - `params`: Binary decisions for each item, e.g., `{"x0": 1, "x1": 0, ...}`
  - Returns: `float` - Total value of selected items
- `evaluate_constraints(params: dict[str, int])`: Evaluate constraint violations.
  - Args:
    - `params`: Binary decisions for each item, e.g., `{"x0": 1, "x1": 0, ...}`
  - Returns: `list[float]` - Constraint values (should be >= 0 for feasible solutions)

#### Instance Properties

- `values`: List of item values
- `weights`: 2D list of item weights in each dimension
- `capacities`: List of capacity constraints for each dimension
- `n_items`: Number of items
- `n_dimensions`: Number of dimensions

## Installation

This benchmark uses only Python standard library and has no external dependencies.
