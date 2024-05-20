---
author: 'Optuna team'
title: 'Simulated Annealing Sampler'
description: 'Sampler based on simulated annealing algorithm.'
tags: ['sampler', 'simulated annealing']
optuna_versions: ['3.5.0', '3.6.0']
license: 'MIT License'
---

## Class or Function Names
- SimulatedAnnealingSampler

## Example
```python
mod = optunahub.load_module("samplers/simulated_annealing")
sampler = mod.SimulatedAnnealingSampler()
```
See `example.py` for more details.

## Others
This package provides a sampler based on Simulated Annealing algorithm.
For more details, see [the documentation](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/005_user_defined_sampler.html).
