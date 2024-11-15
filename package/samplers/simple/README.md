---
author: Optuna team
title: Simple Sampler
description: An easy sampler base class to implement custom samplers.
tags: [sampler, development]
optuna_versions: [3.6.1]
license: MIT License
---

`SimpleBaseSampler` has been moved to [`optunahub.samplers`](https://optuna.github.io/optunahub/samplers.html). Please use [`optunahub.samplers.SimpleBaseSampler`](https://optuna.github.io/optunahub/generated/optunahub.samplers.SimpleBaseSampler.html#optunahub.samplers.SimpleBaseSampler) instead of this package.

## Class or Function Names

- SimpleBaseSampler

## Example

```python
import optunahub

class UserDefinedSampler(
    optunahub.samplers.SimpleBaseSampler
):
    ...
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/simple/example.py) for more details.

## Others

This package provides an easy sampler base class to implement custom samplers.
You can make your own sampler easily by inheriting `SimpleBaseSampler` and by implementing necessary methods.
