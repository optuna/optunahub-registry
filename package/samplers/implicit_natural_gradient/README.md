---
author: Yuhei Otomo and Masashi Shibata
title: Implicit Natural Gradient Sampler
description: A sampler based on Implicit Natural Gradient.
tags: [sampler]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- ImplicitNaturalGradientSampler

## Example

```python
mod = optunahub.load_module("samplers/implicit_natural_gradient")
sampler = mod.ImplicitNaturalGradientSampler()
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/implicit_natural_gradient/example.py) for more details.

## Others

### Reference

Yueming Lyu, Ivor W. Tsang (2019). Black-box Optimizer with Implicit Natural Gradient. arXiv:1910.04301
