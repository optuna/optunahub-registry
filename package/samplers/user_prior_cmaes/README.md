---
author: Shuhei Watanabe
title: CMA-ES with User Prior
description: You can provide the initial parameters, i.e. mean vector and covariance matrix, for CMA-ES with this sampler.
tags: [sampler, cma-es, meta-learning, prior]
optuna_versions: [4.0.0]
license: MIT License
---

## Abstract

As the Optuna CMA-ES sampler does not support any flexible ways to initialize the parameters of the Gaussian distribution, so I created a workaround to do so.

## Class or Function Names

- UserPriorCmaEsSampler

## Installation

```shell
$ pip install optunahub cmaes
```

## Example

```python
import optuna
import optunahub


def objective(trial):
  x = trial.suggest_float("x", -5, 5)
  return x**2


sampler = optunahub.load_module(package="samplers/gp").GPSampler()
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)
```

## Others

Please fill in any other information if you have here by adding child sections (###).
If there is no additional information, **this section can be removed**.

<!--
For example, you can add sections to introduce a corresponding paper.

### Reference
Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019.
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.

### Bibtex
```
@inproceedings{optuna_2019,
    title={Optuna: A Next-generation Hyperparameter Optimization Framework},
    author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
    booktitle={Proceedings of the 25th {ACM} {SIGKDD} International Conference on Knowledge Discovery and Data Mining},
    year={2019}
}
```
-->
