---
author: Kenshin Abe
title: Ensembled Sampler
description: A sampler that ensembles multiple samplers.
tags: [sampler, ensemble]
optuna_versions: [3.6.1]
license: MIT License
---

<!--
This is an example of the frontmatters.
All columns must be string.
You can omit quotes when value types are not ambiguous.
For tags, a package placed in
- package/samplers/ must include the tag "sampler"
- package/visualilzation/ must include the tag "visualization"
- package/pruners/ must include the tag "pruner"
respectively.

---
author: Optuna team
title: My Sampler
description: A description for My Sampler.
tags: [sampler, 2nd tag for My Sampler, 3rd tag for My Sampler]
optuna_versions: [3.6.1]
license: "MIT License"
---
-->

## Installation

No additional packages are required.

## Abstract

This package provides a sampler that ensembles multiple samplers.
You can specify the list of samplers to be ensembled.

## Class or Function Names

- EnsembledSampler

## Example

```python
import optuna
import optunahub

mod = optunahub.load_module("samplers/ensembled")

samplers = [
    optuna.samplers.RandomSampler(),
    optuna.samplers.TPESampler(),
    optuna.samplers.CmaEsSampler(),
]
sampler = mod.EnsembledSampler(samplers)
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/ensembled/example.py) for more details.
