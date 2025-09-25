---
author: Jacob Pfeil
title: Optuna Wrap of CatCMA with Margin [Hamano et al. 2025]
description:
tags: [sampler, cmaes, mixed-variable optimization]
optuna_versions: [4.5.0]
license: MIT License
---

## Abstract

CatCMA with Margin \[Hamano et al. 2025\]
CatCMA with Margin (CatCMAwM) is a method for mixed-variable optimization problems, simultaneously optimizing continuous, integer, and categorical variables. CatCMAwM extends CatCMA by introducing a novel integer handling mechanism, and supports arbitrary combinations of continuous, integer, and categorical variables in a unified framework. \[https://github.com/CyberAgentAILab/cmaes\]

## APIs

CatCmawmSampler

## Example

```python
from __future__ import annotations

import numpy as np
import optuna
import optunahub


def SphereIntCOM(x: np.ndarray, z: np.ndarray, c: np.ndarray) -> float:
    return sum(x * x) + sum(z * z) + len(c) - sum(c[:, 0])


def objective(trial: optuna.Trial) -> float:
    x1 = trial.suggest_float("x1", -5, 5)
    x2 = trial.suggest_float("x2", -5, 5)

    z1 = trial.suggest_int("z1", -1, 1)
    z2 = trial.suggest_int("z2", -2, 2)

    c1 = trial.suggest_categorical("c1", [0, 1, 2])
    c2 = trial.suggest_categorical("c2", [0, 1, 2])

    return SphereIntCOM(
        np.array([x1, x2]).reshape(-1, 1),
        np.array([z1, z2]).reshape(-1, 1),
        np.array([c1, c2]).reshape(-1, 1),
    )
    
    
module = optunahub.load_module(
        package="samplers/catcmawm",
    ) 

study = optuna.create_study(sampler=module.CatCmawmSampler())
study.optimize(objective, n_trials=20)
print(study.best_params)
```
