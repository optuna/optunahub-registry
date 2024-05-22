---
author: 'Optuna team'
title: 'Demo Sampler'
description: 'Demo Sampler of OptunaHub'
tags: ['sampler']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- DemoSampler

## Example
```python
module = optunahub.load_module("samplers/demo")
sampler = module.DemoSampler(seed=42)
```
See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/demo/example.py) for more details.
