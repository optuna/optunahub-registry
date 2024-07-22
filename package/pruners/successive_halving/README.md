---
author: Optuna team
title: Successive Halving Pruner
description: Pruner using Asynchronous Successive Halving Algorithm (ASHA).
tags: [pruner, built-in]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- SuccessiveHalvingPruner

## Example

```python
study = optuna.create_study(
    direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner()
)
study.optimize(objective, n_trials=20)
```

See [example.py](https://github.com/optuna/optunahub-registry/blob/main/package/pruners/successive_halving/example.py) for a full example.

## Others

See the [documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html) for more details.
