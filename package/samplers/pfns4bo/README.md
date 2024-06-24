---
author: "HideakiImamura"
title: "PFNs4BO sampler"
description: "In-context learning for Bayesian optimization. This sampler uses Prior-data Fitted Networks (PFNs) as a surrogate model for Bayesian optimization."
tags: ["sampler"]
optuna_versions: ["3.6.1"]
license: "MIT License"
---

## Class or Function Names
- PFNs4BOSampler

## Installation
```bash
pip install -r requirements.txt
```

## Example
```python
from __future__ import annotations

import os

import optuna
import optunahub


module = optunahub.load_module("samplers/pfns4bo")
PFNs4BOSampler = module.PFNs4BOSampler


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    study = optuna.create_study(
        sampler=PFNs4BOSampler(),
    )
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)
```

## Others

### Reference

Samuel Müller, Matthias Feurer, Noah Hollmann, and Frank Hutter. 2023. PFNs4BO: in-context learning for Bayesian optimization. In Proceedings of the 40th International Conference on Machine Learning (ICML'23), Vol. 202. JMLR.org, Article 1056, 25444–25470.
