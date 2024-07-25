---
author: Optuna team
title: Threshold Pruner
description: Pruner to detect outlying metrics of the trials.
tags: [pruner, built-in]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- ThresholdPruner

## Example

```python
study = create_study(pruner=ThresholdPruner(upper=1.0))
study.optimize(objective_for_upper, n_trials=10)

study = create_study(pruner=ThresholdPruner(lower=0.0))
study.optimize(objective_for_lower, n_trials=10)
```

See [example.py](https://github.com/optuna/optunahub-registry/blob/main/package/pruners/threshold/example.py) for a full example.

## Others

See the [documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.ThresholdPruner.html) for more details.
