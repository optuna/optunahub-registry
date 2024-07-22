---
author: Optuna team
title: Wilcoxon Pruner
description: Pruner based on the Wilcoxon signed-rank test.
tags: [pruner, built-in]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- WilcoxonPruner

## Example

```python
study = optuna.create_study(pruner=optuna.pruners.WilcoxonPruner(p_threshold=0.1))
study.optimize(objective, n_trials=100)
```

## Others

See the [documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.WilcoxonPruner.html) for more details.
