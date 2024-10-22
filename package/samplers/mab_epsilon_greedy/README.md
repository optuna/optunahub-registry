---
author: Ryota Nishijima
title: Sampler Using Multi-Aarmed Bandit Epsilon-Greedy Algorithm
description: Sampler based on multi-armed bandit algorithm with epsilon-greedy arm selection.
tags: [sampler, multi-armed bandit]
optuna_versions: [4.0.0]
license: MIT License
---

## Class or Function Names

- MABEpsilonGreedySampler

## Example

```python
mod = optunahub.load_module("samplers/mab_epsilon_greedy")
sampler = mod.MABEpsilonGreedySampler()
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/mab_epsilon_greedy/example.py) for more details.

## Others

This package provides a sampler based on Multi-armed bandit algorithm with epsilon-greedy selection.
