---
author: Optuna Team
title: Constrained robust Bayesian optimization of expensive noisyblack-box functions with guaranteed regret bounds (CARBO)
description: Constrained Bayesian optimization with input uncertainties
tags: [sampler, bo, bayesian-optimization, constrained-optimization, carbo]
optuna_versions: [4.4.0]
license: MIT License
---

## Abstract

This package implements a modified Constrained Adversarially Robust Bayesian Optimization (CARBO) sampler based on [the paper `Constrained robust Bayesian optimization of expensive noisyblack-box functions with guaranteed regret bounds`](https://aiche.onlinelibrary.wiley.com/doi/epdf/10.1002/aic.17857).
This sampler robustly optimizes a function along with inequality constraints that incurs a noise in its input.
The algorithm details are described in the `Others` section.

## APIs

- `CARBOSampler(*, seed: int | None = None, independent_sampler: BaseSampler | None = None, n_startup_trials: int = 10, deterministic_objective: bool = False, constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None, rho: float = 1e3, beta: float = 4.0, input_noise_rads = {}, n_local_search: int = 10)`
  - `seed`: Seed for random number generator.
  - `independent_sampler`: Sampler used for initial sampling (for the first `n_startup_trials` trials) and for conditional parameters. (a random sampler with the same `seed` is used).
    Sampler used when `sample_independent` is called.
  - `n_startup_trials`: Number of initial trials.
  - `deterministic_objective`: Whether the objective function is deterministic or not. If `True`, the sampler will fix the noise variance of the surrogate model to the minimum value (slightly above 0 to ensure numerical stability).
  - `constraints_func`: An optional function that computes the objective constraints. It must take a `optuna.trial.FrozenTrial` and return the constraints. The return value must be a sequence of `float` s. A value strictly larger than 0 means that a constraints is violated. A value equal to or smaller than 0 is considered feasible. If `constraints_func` returns more than one value for a trial, that trial is considered feasible if and only if all values are equal to 0 or smaller. The `constraints_func` will be evaluated after each successful trial. The function won't be called when trials fail or are pruned, but this behavior is subject to change in future releases. Currently, the `constraints_func` option is not supported for multi-objective optimization.
  - `rho`: The mix up coefficient for the acquisition function. If this value is large, the parameter suggestion puts more priority on constraints.
  - `beta`: The coefficient for LCB and UCB. If this value is large, the parameter suggestion becomes more pessimistic, meaning that the search is inclined to explore more.
  - `input_noise_rads`: The input noise ranges for each parameter. For example, when `{"x": 0.1, "y": 0.2}`, the sampler assumes that +/- 0.1 is acceptable for `x` and +/- 0.2 is acceptable for `y`. This determines `W(theta)`.
  - `const_noisy_param_names`: The list of parameters determined externally rather than being decision variables. For these parameters, `suggest_float` returns values that are adversally determined by the environement instead of searching values that optimize the objective function.
  - `n_local_search`: How many times the local search is performed.
  - `nominal_ranges`: An optional dictionary to override nominal ranges for a subset of parameters. If a range is specified for a parmaeter, it's nominal value is sampled from the given range instead of the range specified to `suggest_float`. This option is useful for avoiding clipping: if the noise range is +/- eps, specify \[L, U\] as a nominal range and specify \[L-eps, U+eps\] for `suggest_float`.

Note that because of the limitation of the algorithm, only non-conditional numerical parameters can be sampled by the MO-CMA-ES algorithm, and categorical and conditional parameters are handled by random search.

## Installation

```shell
$ pip install torch --index-url https://download.pytorch.org/whl/cpu
$ pip install scipy
```

## Example

```python
import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 0.0, 2 * np.pi)
    y = trial.suggest_float("y", 0.0, 2 * np.pi)
    c = float(np.sin(x) * np.sin(y) + 0.95)
    trial.set_user_attr("c", c)
    return float(np.sin(x) + y)


def constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    c = trial.user_attrs["c"]
    return (c, )


CARBOSampler = optunahub.load_module("samplers/carbo").CARBOSampler
sampler = CARBOSampler(seed=0, constraints_func=constraints)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=30)

```

## Others

Please look at [GitHub README.md](https://github.com/optuna/optunahub-registry/tree/main/package/samplers/carbo#others) for the compiled version.

### Notations

In this section, we use the following notations:

- $x \in [0, 1]^D$, an input vector,
- $B_\epsilon \coloneqq [-\frac{\epsilon}{2}, \frac{\epsilon}{2}]^D$, an $\epsilon$-bounding box,
- $\xi \in B_\epsilon$, an input noise,
- $f: [0, 1]^D \rightarrow \mathbb{R}$, an objective function,
- $g_c: [0, 1]^D \rightarrow \mathbb{R}$, the $c$-th constraint,
- $\text{LCB}_{h}: [0, 1]^D \rightarrow \mathbb{R}$, the lower confidence bound of a function $h$,
- $\text{UCB}_{h}: [0, 1]^D \rightarrow \mathbb{R}$, the upper confidence bound of a function $h$.

Please note that both $g_c$ and $f$ are standardized internally so that their distributions follow the assumption on the prior distribution by the Gaussian process.

Suppose we would like to solve the following max-min problem:
$\max_{x \in [0, 1]^D} \min_{\xi \in B_\epsilon} f(x + \xi) \text{ subject to } g_c(x + \xi) \geq 0 \; (\text{for }c \in {1, 2, \dots, C}).$
where the actual input noise $\xi$ is assumed to be drawn from $B_\epsilon$ uniformly.

### Algorithm Details

1. Train Gaussian process regressors for each function $f, g_1, \dots, g_C$ using the past observations.
1. Solve the following max-min problem:
   $x_{\star} \in \text{arg}\max_{x \in [0, 1]^D}\min_{\xi \in B_\epsilon} \text{UCB}_{f}(x + \xi) + \rho \sum_{c = 1}^C [\text{UCB}_{g_c}(x + \xi)]^{-}$ where $[a]^{-} \coloneqq \min(0, a)$.
1. Solve the following minimization problem:
   $\xi_{\star} \in \text{arg}\min_{\xi \in B_\epsilon} \text{LCB}_{f}(x_\star + \xi) + \rho\sum_{c = 1}^C [\text{LCB}_{g_c}(x_\star + \xi)]^{-}$
1. Evaluate each function at $x = x_{\star} + \xi_{\star}$.
1. Go back to 1.

In principle, $[\text{UCB}_{g_c}(x + \xi)]^{-}$ and $[\text{LCB}_{g_c}(x + \xi)]^{-}$ quantify the upper and lower confidence bounds of the violation amount.
Please note that Processes 2 and 3 are modified from the original paper because our setup assumes that the same input noise $\xi$ is used for each constraint and the objective evaluations.
Also, the order of the min or max operation and the summation is flipped in our implementation.
