---
author: HirokiTakizawa
title: HEBO (Heteroscedastic and Evolutionary Bayesian Optimisation) supporting Define-by-Run and parallelization
description: This package offers HEBO algorithm using BaseSampler and supports parallelization in exchange for increased computation.
tags: [sampler, Bayesian optimization, Heteroscedastic Gaussian process, Evolutionary algorithm]
optuna_versions: [4.1.0]
license: MIT License
---

## Class or Function Names

- HEBOSampler

## Installation

```bash
pip install -r https://hub.optuna.org/samplers/hebo_base_sampler/requirements.txt
git clone git@github.com:huawei-noah/HEBO.git
cd HEBO/HEBO
pip install -e .
```

## Example

```python
def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    y = trial.suggest_int("y", -1, 1)
    return x ** 2 + y
sampler = HEBOSampler(constant_liar=True)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=20, n_jobs=2)
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/hebo_base_sampler/example.py) for a full example.

## Others

This package is based on [the preceding HEBO package](https://hub.optuna.org/samplers/hebo/) authored by HideakiImamura.

HEBO is the winning submission to the [NeurIPS 2020 Black-Box Optimisation Challenge](https://bbochallenge.com/leaderboard).
Please refer to [the official repository of HEBO](https://github.com/huawei-noah/HEBO/tree/master/HEBO) for more details.

### Reference

Cowen-Rivers, Alexander I., et al. "An Empirical Study of Assumptions in Bayesian Optimisation." arXiv preprint arXiv:2012.03826 (2021).
