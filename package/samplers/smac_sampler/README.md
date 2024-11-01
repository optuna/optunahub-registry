---
author: Difan Deng
title: SMAC3
description: SMAC offers a robust and flexible framework for Bayesian Optimization to support users in determining well-performing hyperparameter configurations for their (Machine Learning) algorithms, datasets and applications at hand. The main core consists of Bayesian Optimization in combination with an aggressive racing mechanism to efficiently decide which of two configurations performs better.
tags: [sampler, Bayesian optimization, Gaussian process, Random Forest]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- SAMCSampler

## Installation

```bash
pip install -r https://hub.optuna.org/samplers/hebo/requirements.txt
pip install smac==2.2.0
```

## Example

```python
search_space = {
    "x": FloatDistribution(-10, 10),
    "y": IntDistribution(0, 10),

}
sampler = SMACSampler(search_space)
study = optuna.create_study(sampler=sampler)
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/hebo/example.py) for a full example.
![History Plot](images/smac_optimization_history.png "History Plot")

## Others

SMAC is maintained by the SMAC team in [automl.org](https://www.automl.org/). If you have trouble using SMAC, a concrete question or found a bug, please create an issue under the [SMAC](https://github.com/automl/SMAC3) repository.

For all other inquiries, please write an email to smac\[at\]ai\[dot\]uni\[dash\]hannover\[dot\]de.

### Reference

Lindauer et al. "SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter Optimization", Journal of Machine Learning Research, http://jmlr.org/papers/v23/21-0888.html
