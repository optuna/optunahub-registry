---
author: Optuna team
title: Patient Pruner
description: Pruner which wraps another pruner with tolerance.
tags: [pruner, built-in]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- PatientPruner

## Example

```python
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.PatientPruner(optuna.pruners.MedianPruner(), patience=1),
)
study.optimize(objective, n_trials=20)
```

See [example.py](https://github.com/optuna/optunahub-registry/blob/main/package/pruners/patient/example.py) for a full example.

## Others

See the [documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PatientPruner.html) for more details.
