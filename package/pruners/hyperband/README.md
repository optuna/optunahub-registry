---
author: Optuna team
title: Hyperband Pruner
description: Pruner using Hyperband.
tags: [pruner, built-in]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- HyperbandPruner

## Example

```python
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=n_train_iter, reduction_factor=3
    ),
)
study.optimize(objective, n_trials=20)
```

See [example.py](https://github.com/optuna/optunahub-registry/blob/main/package/pruners/hyperband/example.py) for a full example.

## Others

See the [documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html) for more details.
