---
author: 'Optuna team'
title: 'BoTorch Sampler'
description: 'A Sampler using botorch library as the backend.'
tags: ['sampler', 'built-in']
optuna_versions: ['3.6.1']
license: 'MIT License'
---

## Class or Function Names
- BoTorchSampler

## Installation
```bash
pip install optuna-integration botorch
```

## Example
```python
module = optunahub.load_module("samplers/botorch_sampler")
sampler = module.BoTorchSampler()
```

## Others
See the [documentation](https://optuna-integration.readthedocs.io/en/latest/reference/generated/optuna_integration.BoTorchSampler.html) for more details.

