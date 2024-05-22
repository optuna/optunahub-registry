---
author: 'Optuna team'
title: 'Plot Hypervolume History with Reference Point'
description: 'Plot hypervolume history with the reference point information.'
tags: ['visualization', 'hypervolume', 'multi-objective optimization']
optuna_versions: ['3.6.0']
license: 'MIT License'
---

## Class or Function Names
- plot_hypervolume_history

## Example
```python
mod = optunahub.load_module("visualization/plot_hypervolume_history_with_rp")
mod.plot_hypervolume_history(study, reference_point)
```
See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/visualization/plot_hypervolume_history_with_rp/example.py) for more details.
The example of generated image is as follows.

![Example](images/example.png "Example")
