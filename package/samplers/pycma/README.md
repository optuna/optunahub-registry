---
author: 'Optuna team'
title: 'PyCMA Sampler'
description: 'A CMA-ES Sampler using cma library as the backend.'
tags: ['sampler', 'built-in']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- PyCmaSampler

## Installation
```bash
pip install optuna-integration cma
```

## Example
```python
module = optunahub.load_module("samplers/pycma")
sampler = module.PyCmaSampler()
```

## Others
See the [documentation](https://optuna-integration.readthedocs.io/en/latest/reference/generated/optuna_integration.PyCmaSampler.html) for more details.
