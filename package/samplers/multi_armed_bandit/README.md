---
author: Ryota Nishijima
title: Multi-armed Bandit Sampler
description: Sampler based on multi-armed bandit algorithm with epsilon-greedy arm selection.
tags: [sampler, multi-armed bandit]
optuna_versions: [4.0.0]
license: MIT License
---

## Class or Function Names

- MultiArmedBanditSampler

## Example

```python
mod = optunahub.load_module("samplers/multi_armed_bandit")
sampler = mod.MultiArmedBanditSampler()
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/multi_armed_bandit/example.py) for more details.

## Others

This package provides a sampler based on Multi-armed bandit algorithm with epsilon-greedy selection.
