---
author: 'Optuna team'
title: 'Simple Sampler'
description: 'An easy sampler base class to implement custom samplers.'
tags: ['sampler', 'development']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- SimpleSampler

## Example
```python
class UserDefinedSampler(
    optunahub.load_module("samplers/simple").SimpleSampler
):
    ...
```
See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/simple/example.py) for more details.

## Others
This package provides an easy sampler base class to implement custom samplers.
You can make your own sampler easily by inheriting `SimpleSampler` and by implementing necessary methods.
