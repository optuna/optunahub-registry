---
author: Optuna team
title: Simulated Annealing Sampler
description: Sampler based on simulated annealing algorithm.
tags: [sampler, simulated annealing]
optuna_versions: [3.5.0, 3.6.0]
license: MIT License
---

## Class or Function Names

- SimulatedAnnealingSampler

## Example

```python
mod = optunahub.load_module("samplers/simulated_annealing")
sampler = mod.SimulatedAnnealingSampler()
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/simulated_annealing/example.py) for more details.
You can run the [example in Google Colab](http://colab.research.google.com/github/optuna/optunahub-registry/blob/main/package/samplers/simulated_annealing/example.ipynb).

## Others

This package provides a sampler based on Simulated Annealing algorithm.
For more details, see [the documentation](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/005_user_defined_sampler.html).
