---
author: Ryota Ozaki
title: PLMBO (Preference Learning Multi-Objective Bayesian Optimization)
description: Interatctive multi-objective Bayesian optimization based on user preference
tags: [sampler, interactive optimization, Bayesian optimization, multi-objective optimization, preference learning, active learning]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- PLMBOSampler

## Installation

```sh
pip install -r https://hub.optuna.org/samplers/plmbo/requirements.txt
```

## Example

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import optuna
import optunahub
from optuna.distributions import FloatDistribution
import numpy as np


PLMBOSampler = optunahub.load_module(  # type: ignore
    "samplers/plmbo",
).PLMBOSampler

if __name__ == "__main__":
    f_sigma = 0.01

    def obj_func1(x):
        return np.sin(x[0]) + x[1]

    def obj_func2(x):
        return -np.sin(x[0]) - x[1] + 0.1

    def obs_obj_func(x):
        return np.array(
            [
                obj_func1(x) + np.random.normal(0, f_sigma),
                obj_func2(x) + np.random.normal(0, f_sigma),
            ]
        )

    def objective(trial: optuna.Trial):
        x1 = trial.suggest_float("x1", 0, 1)
        x2 = trial.suggest_float("x2", 0, 1)
        values = obs_obj_func(np.array([x1, x2]))
        return float(values[0]), float(values[1])

    sampler = PLMBOSampler(
        {
            "x1": FloatDistribution(0, 1),
            "x2": FloatDistribution(0, 1),
        }
    )
    study = optuna.create_study(sampler=sampler, directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=20)

    optuna.visualization.matplotlib.plot_pareto_front(study)
    plt.show()
```

## Others

### Reference

R Ozaki, K Ishikawa, Y Kanzaki, S Takeno, I Takeuchi, and M Karasuyama. (2024). Multi-Objective Bayesian Optimization with Active Preference Learning. Proceedings of the AAAI Conference on Artificial Intelligence.

### Bibtex

```
@inproceedings{ozaki2024multi,
  title={Multi-Objective Bayesian Optimization with Active Preference Learning},
  author={Ozaki, Ryota and Ishikawa, Kazuki and Kanzaki, Youhei and Takeno, Shion and Takeuchi, Ichiro and Karasuyama, Masayuki},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={13},
  pages={14490--14498},
  year={2024}
}
```
