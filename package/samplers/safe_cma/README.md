---
author: Optuna Team
title: SafeCMA Sampler
description: A sampler using SafeCMA, a variant of CMA-ES that incorporates safety constraints for safe optimization.
tags: [sampler, CMA-ES, Constrained optimization, Safe optimization]
optuna_versions: [4.6]
license: MIT License
---

## Abstract

SafeCMASampler provides an implementation of SafeCMA, a variant of CMA-ES that incorporates safety constraints. This sampler extends the standard CMA-ES algorithm to handle constrained optimization problems where certain regions of the search space must be avoided. SafeCMA uses Gaussian Process models to estimate Lipschitz constants and manage trust regions, ensuring that the optimization process respects safety constraints while exploring the search space efficiently.

## Class or Function Names

- `SafeCMASampler(safe_seeds: Sequence[Sequence[float]], seeds_evals: Sequence[float], seeds_safe_evals: Sequence[float] | Sequence[Sequence[float]], safety_threshold: Sequence[float], safe_function: Callable[[Sequence[float | int]], float], sigma0: float | None = None, seed: int | None = None, independent_sampler: BaseSampler | None = None, warn_independent_sampling: bool = True, *, popsize: int | None = None, n_max_resampling: int = 100, cov: Sequence[Sequence[float]] | None = None)`
  - `safe_seeds`: Initial set of safe seed points given to SafeCMA (required, together with `seeds_evals` and `seeds_safe_evals`). Should be a list of lists or a 2D array with shape `(n_seeds, n_dimensions)`.
  - `seeds_evals`: Objective function values for the safe seeds (required if `safe_seeds` is specified). Should be a list or array of floats with shape `(n_seeds,)`.
  - `seeds_safe_evals`: Safety function values for the safe seeds (required if `safe_seeds` is specified). Should be a list or array of floats with shape `(n_seeds, 1)` or `(n_seeds,)`.
  - `safety_threshold`: Threshold for what is considered "safe". Values from `safe_function` should be less than or equal to this. Should be a list or array of floats.
  - `safe_function`: Safety function. A function that takes a sequence (list/array) of parameter values and returns a safety value. The parameters are given in the order defined by the search space. The sampler calls this function for each trial to determine safety.
  - `sigma0`: Initial standard deviation for the optimizer. If not specified, it defaults to `min_range / 6`, where `min_range` is the smallest width in the search space.
  - seed: Random seed.
  - `independent_sampler`: An Optuna sampler instance used for parameters not in the CMA-ES search space (for example, categorical/conditional parameters). If not specified, `optuna.samplers.RandomSampler` is used by default.
  - warn_independent_sampling: If `True`, print a warning when independent sampler is used (the first trial in a study always uses the independent sampler, but warning is skipped in that case).
  - `popsize`: Population size for SafeCMA.
  - `n_max_resampling`: Maximum number of resamples per parameter (default: 100).
  - `cov`: Covariance matrix (optional).

Note that because of the limitation of the algorithm, only non-conditional numerical parameters can be sampled by the SafeCMA algorithm, and categorical and conditional parameters are handled by the independent sampler.

## Installation

```bash
pip install -r https://hub.optuna.org/samplers/safe_cma/requirements.txt
```

## Example

```python
from __future__ import annotations

from typing import Sequence

import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


def safe_function(x: Sequence[float]) -> float:
    # Example safety function: x[0] must be <= 0 to be safe
    return x[0]


if __name__ == "__main__":
    # Generate initial safe seeds
    safe_seeds = [[-2.0, -2.0], [-1.0, -1.0], [-0.5, -0.5]]
    seeds_evals = [x**2 + y**2 for x, y in safe_seeds]
    seeds_safe_evals = [[safe_function([x, y])] for x, y in safe_seeds]
    safety_threshold = [0.0]

    sampler = optunahub.load_module(
        package="samplers/safe_cma",
    ).SafeCMASampler(
        safe_seeds=safe_seeds,
        seeds_evals=seeds_evals,
        seeds_safe_evals=seeds_safe_evals,
        safety_threshold=safety_threshold,
        safe_function=safe_function,
        sigma0=1.0,
        seed=42,
    )

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)
    print(study.best_params)
```

### Reference

Kento Uchida, Ryoki Hamano, Masahiro Nomura, Shota Saito, and Shinichi Shirakawa. 2024. CMA-ES for Safe Optimization. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '24). Association for Computing Machinery, New York, NY, USA, 722–730. https://doi.org/10.1145/3638529.3654193

### Bibtex

```
@inproceedings{uchida2024cma,
  title={CMA-ES for Safe Optimization},
  author={Uchida, Kento and Hamano, Ryoki and Nomura, Masahiro and Saito, Shota and Shirakawa, Shinichi},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  pages={722--730},
  year={2024}
}
```
