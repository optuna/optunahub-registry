---
author: Optuna Team
title: MAPCMA sampler
description: A sampler that extends the CMA-ES rank-one-update based on the MAP-IGO (maximum a posteriori information geometric optimization) framework.
tags: [sampler, CMA-ES, Evolutionary algorithms]
optuna_versions: [4.6]
license: MIT License
---

## Abstract

MAPCMASampler provides an implementation of the MAP-IGO (maximum a posteriori information geometric optimization) framework, which extends the CMA-ES rank-one-update. This sampler adds momentum-based updates to the standard CMA-ES, following the MAP-IGO algorithm.

## Class or Function Names

- `MAPCMASampler(mean: dict[str, Any] | None = None, sigma0: float | None = None, seed: int | None = None, popsize: int | None = None, cov: np.ndarray | None = None, momentum_r: float | None = None, search_space: dict[str, BaseDistribution] | None = None, independent_sampler: BaseSampler | None = None)`
  - `mean`: Initial mean of MAPCMA. If not provided, the mean will be set to the center of the search space.
  - `sigma0`: The initial standard deviation of covariance matrix. If not provided, it will be set based on the search space bounds.
  - `seed`: The seed of the random number generator.
  - `popsize`: The population size. If not provided, the population size will be set based on the search space dimensionality (4 + floor(3 * log(dimension))).
  - `cov`: The initial covariance matrix. If not provided, it will be set to the identity matrix scaled by sigma0.
  - `momentum_r`: Scaling ratio of momentum update. This parameter controls the momentum-based update in the MAP-IGO framework.
  - `search_space`: A dictionary containing the search space that defines the parameter space. The keys are the parameter names and the values are [the parameter's distribution](https://optuna.readthedocs.io/en/stable/reference/distributions.html). If the search space is not provided, the sampler will infer the search space dynamically during the first trial.
  - `independent_sampler`: A :class:`~optuna.samplers.BaseSampler` instance that is used for independent sampling. The parameters not contained in the relative search space are sampled by this sampler. If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler` is used as the default.

Note that because of the limitation of the algorithm, only non-conditional numerical parameters can be sampled by the MAPCMA algorithm, and categorical and conditional parameters are handled by the independent sampler.

## Installation

```bash
pip install -r https://hub.optuna.org/samplers/mapcma/requirements.txt
```

## Example

```python
from __future__ import annotations

import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


if __name__ == "__main__":
    sampler = optunahub.load_module(
        package="samplers/mapcma",
    ).MAPCMASampler()

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)
    print(study.best_params)
```

### Reference

Hamano, Ryoki, Shinichi Shirakawa, and Masahiro Nomura. "Natural Gradient Interpretation of Rank-One Update in CMA-ES." International Conference on Parallel Problem Solving from Nature. Cham: Springer Nature Switzerland, 2024.

### BibTex

```
@inproceedings{hamano2024natural,
  title={Natural Gradient Interpretation of Rank-One Update in CMA-ES},
  author={Hamano, Ryoki and Shirakawa, Shinichi and Nomura, Masahiro},
  booktitle={International Conference on Parallel Problem Solving from Nature},
  pages={252--267},
  year={2024},
  organization={Springer}
}
```
