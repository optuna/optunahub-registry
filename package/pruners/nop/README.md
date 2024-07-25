---
author: Optuna team
title: Nop Pruner
description: Pruner which never prunes trials.
tags: [pruner, built-in]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- NopPruner

## Example

```python
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
study.optimize(objective, n_trials=20)
```

See [example.py](https://github.com/optuna/optunahub-registry/blob/main/package/pruners/nop/example.py) for a full example.

## Others

See the [documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.NopPruner.html) for more details.
