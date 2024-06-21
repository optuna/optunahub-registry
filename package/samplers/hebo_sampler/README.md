---
author: HideakiImamura
title: HEBO (Heteroscedastic and Evolutionary Bayesian Optimisation)
description: HEBO addresses the problem of noisy and heterogeneous objective functions by using a heteroscedastic Gaussian process and an evolutionary algorithm.
tags: ["sampler"]
optuna_versions: ["3.6.1"]
license: "MIT License" 
---

## Class or Function Names
- HEBOSampler

## Installation
```bash
pip install -r requirements.txt
git clone git@github.com:huawei-noah/HEBO.git
cd HEBO/HEBO
pip install -e .
```

## Example
```python
search_space = {
    "x": FloatDistribution(-10, 10),
    "y": IntDistribution(0, 10),

}
sampler = HEBOSampler(search_space)
study = optuna.create_study(sampler=sampler)
```
See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/hebo_sampler/example.py) for a full example.
![History Plot](images/hebo_optimization_history.png "History Plot")


## Others

### Reference

Cowen-Rivers, Alexander I., et al. "An Empirical Study of Assumptions in Bayesian Optimisation." arXiv preprint arXiv:2012.03826 (2021).
