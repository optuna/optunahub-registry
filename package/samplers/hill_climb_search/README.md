---
author: Chinmaya Sahu
title: Hill Climb Local Search Sampler
description: This sampler used the Hill Climb Algorithm to improve the searching, by selecting the best neighbors and moving in that direction.
tags: [sampler, hill climb]
optuna_versions: [4.0.0]
license: MIT License
---

## Abstract

The **hill climbing algorithm** is an optimization technique that iteratively improves a solution by evaluating neighboring solutions in search of a local maximum or minimum. Starting with an initial guess, the algorithm examines nearby "neighbor" solutions, moving to a better neighbor if one is found. This process continues until no improvement is possible, resulting in a locally optimal solution. Hill climbing is efficient and easy to implement but can get stuck in local optima, making it suitable for simple optimization landscapes or applications with limited time constraints. Variants like random restarts and stochastic selection help overcome some limitations.

## Class or Function Names

- HillClimbSearch

## Example

```python
import optuna
import optunahub
   
def objective(trial):
    x = trial.suggest_discrete_uniform("x", -10, 10)
    y = trial.suggest_discrete_uniform("y", -10, 10)
    return -(x**2 + y**2)

mod = optunahub.load_module("samplers/hill_climb_search")
sampler = mod.HillClimbSearch()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=20)
```
