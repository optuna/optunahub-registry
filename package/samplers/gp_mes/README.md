---
author: Ahmed Eldeeb
title: Max-value Entropy Search Sampler
description: A Gaussian-process sampler using the max-value entropy search (MES) acquisition function, which selects the candidate expected to be most informative about the maximum value of the objective.
tags: [sampler, Bayesian optimization, Gaussian process, acquisition function, entropy search, information gain]
optuna_versions: [4.9.0]
license: MIT License
---

## Class or Function Names

- `MESSampler`

## Installation

```shell
pip install scipy torch
```

## Overview

Optuna's acquisition functions all score a candidate by improvement over the incumbent
(`LogEI`, `LogPI`) or by a posterior quantile (`UCB`, `LCB`). Max-value entropy search scores
it instead by *information*: how much observing `f(x)` is expected to reduce the entropy of
the distribution of the maximum value `y* = max_x f(x)`.

Writing `mu(x)`, `sigma(x)` for the posterior mean and standard deviation, `phi` and `Psi` for
the standard normal PDF and CDF, and `gamma = (y* - mu(x)) / sigma(x)`, the acquisition is

```
alpha(x) = mean over y* in Y* of [ gamma * phi(gamma) / (2 * Psi(gamma)) - log Psi(gamma) ]
```

This is the mutual information `I({x, f(x)}; y*)`, obtained by treating `f(x)` as
truncated-normal given `y*` (Wang & Jegelka 2017, eq. 6).

Two consequences follow from using `y*` rather than the location `x*`. The information gain
is one-dimensional, so no expectation propagation is needed, which is what makes predictive
entropy search expensive and numerically delicate. And there is no exploration parameter:
unlike `UCB`'s `beta`, nothing needs tuning.

`MESSampler` subclasses `optuna.samplers.GPSampler` and reuses its kernel fitting,
search-space handling, and acquisition optimizer. Multi-objective and constrained studies fall
back to the parent sampler.

## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2


sampler = optunahub.load_module("samplers/gp_mes").MESSampler(seed=42)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=50)
print(study.best_params)
```

See `example.py` for both max-value samplers.

## API Reference

### `MESSampler`

| Argument               | Type  | Default       | Description                                                                                                                                     |
| ---------------------- | ----- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_value_sampler`    | `str` | `"posterior"` | How to sample the maxima `Y*`. `"posterior"` draws joint posterior paths (section 3.2); `"gumbel"` uses the Gumbel approximation (section 3.1). |
| `n_max_value_samples`  | `int` | `32`          | Number of maxima the acquisition is averaged over.                                                                                              |
| `n_representer_points` | `int` | `512`         | Size of the grid used to sample the maxima.                                                                                                     |

All remaining arguments are those of `optuna.samplers.GPSampler`: `seed`,
`independent_sampler`, `n_startup_trials`, `deterministic_objective`, `constraints_func`,
`warn_independent_sampling`.

### Choosing a max-value sampler

`"posterior"` draws `f` jointly on the representer points and takes the maximum of each path.
Optuna's `GPRegressor.posterior(x, joint=True)` returns the full covariance, so this needs no
random-Fourier-feature approximation. It costs one Cholesky factorization of an
`n_representer_points` square matrix per trial.

`"gumbel"` approximates the CDF of the maximum by the independent product
`P(y* < z) ~= prod_i Psi((z - mu_i) / sigma_i)`, matches a Gumbel distribution at two
quantiles, and samples by inverse transform. It needs only marginals and is much cheaper.

The independence assumption has a measurable cost. Because the grid points are in fact
correlated, the product under-states the CDF and biases `y*` upward, and the bias grows with
the grid size while the true maximum saturates. Measured against 20,000 brute-force joint
posterior draws on a fitted 2-D GP:

| Representer points | Brute-force mean `y*` | Gumbel mean `y*` |     Bias |
| -----------------: | --------------------: | ---------------: | -------: |
|                 25 |                2.6116 |           2.7451 | +0.49 sd |
|                100 |                2.7261 |           2.9170 | +0.58 sd |
|                400 |                2.7487 |           3.1452 | +1.22 sd |
|               1600 |                2.7672 |           3.3714 | +1.81 sd |

The `"posterior"` sampler is unbiased on the same test (−0.02 sd at 400 points), as it must
be, being the reference procedure itself.

An upward-biased `y*` makes MES more exploratory, since it raises `gamma` everywhere and
flattens the acquisition's preference for high posterior means. In the benchmark below
`"posterior"` is clearly better at a 40-trial budget and the two are indistinguishable at
160, which is why `"posterior"` is the default despite costing roughly 1.5x more per trial.

## Benchmark

Setup: BBOB functions f1, f8, f10, f15, f21, f24 (one from each of the five BBOB groups,
plus a second weak-structure multimodal function) through `package/benchmarks/bbob`, at
dimensions 2, 5 and 10. Five BBOB instances times three sampler seeds gives 15 paired
replicates per cell; every sampler sees the same 15 pairs. The metric is simple regret,
`best_observed - f_opt`. Confidence intervals are 95% bootstrap intervals over replicates.
All samplers run at library defaults.

### Mean rank, 1 is best

Across all (function, dimension, instance, seed) cells, ranking the same four samplers:

| sampler                  | 40 trials             | 160 trials            |
| ------------------------ | --------------------- | --------------------- |
| `GPUCBSampler` (beta=2)  | 2.29 [2.16, 2.41]     | **2.28** [2.15, 2.41] |
| `MESSampler` (gumbel)    | 2.90 [2.77, 3.03]     | 2.53 [2.40, 2.67]     |
| `GPSampler` (EI)         | **2.20** [2.07, 2.33] | 2.59 [2.46, 2.72]     |
| `MESSampler` (posterior) | 2.61 [2.49, 2.74]     | 2.60 [2.46, 2.74]     |

Two of those changes are significant, in the sense that the intervals do not overlap: EI
gets relatively worse as the budget grows (2.20 to 2.59) and MES with the Gumbel sampler
gets better (2.90 to 2.53). UCB is flat and leads at both budgets.

At 40 trials against a wider field, the full ordering was `GPSampler` 3.05, `GPUCBSampler`
3.13, `MESSampler` (posterior) 3.55, `GPPISampler` 3.65, `MESSampler` (gumbel) 4.02,
`GPTSSampler` 4.83, `RandomSampler` 5.76.

### Mean rank by dimension, 160 trials

| sampler                  |  d=2 |  d=5 |     d=10 |
| ------------------------ | ---: | ---: | -------: |
| `GPSampler` (EI)         | 2.43 | 2.46 |     2.88 |
| `GPUCBSampler`           | 2.34 | 2.39 | **2.10** |
| `MESSampler` (gumbel)    | 2.44 | 2.39 |     2.77 |
| `MESSampler` (posterior) | 2.78 | 2.77 | **2.26** |

### When this sampler helps, and when it does not

- **Helps**: larger budgets and higher dimensions. At 160 trials in 10-D, `MESSampler`
  with posterior sampling ranks second of four and clearly ahead of EI.
- **Does not help**: small budgets. At 40 trials MES was significantly worse than EI on 5
  of 18 (function, dimension) cells and significantly better on 1. The clearest losses are
  on f1 Sphere at every dimension, where the objective is smooth and unimodal and there is
  nothing to learn about the maximum that EI does not already exploit.
- **`GPUCBSampler` beat every sampler tested at both budgets**, including this one. If you
  want the best expected result on BBOB-like problems rather than an information-theoretic
  acquisition specifically, use that instead.

### A caveat on the 40-trial results

With 10 random startup trials out of 40, 7.3% of GP-sampler studies never improved on the
best point found during startup, so on those cells the numbers say nothing about the
acquisition function. It is concentrated where the budget is most obviously too small:
f10 Ellipsoidal at d=5 and d=10 (18% and 14%) and f21 and f24 at d=2 (19% and 21%). This
compresses the differences between samplers at the 40-trial budget and is part of why the
160-trial comparison separates them more clearly.

### Cost

Median wall-clock per 40-trial study, in seconds: EI 1.0-1.5, UCB 0.9-1.4, MES gumbel
1.4-2.0, MES posterior 2.2-3.0, Thompson sampling 3.2-3.9.

### Correctness

The registry does not review scientific validity, so the acquisition was checked three
independent ways rather than asserted:

1. Against numerical quadrature of the truncated-normal entropy, over 60 random
   `(mu, sigma, y*)` triples. Maximum absolute error `1.9e-15`.
1. Against a separate `scipy`/`erfc` evaluation of eq. 6 in the test suite.
1. Against BoTorch's `qMaxValueEntropy`, as a consistency check. The two are not the same
   estimand: BoTorch estimates the *noisy* information gain by Monte Carlo over fantasy
   models, while this package computes the closed form for the noiseless case, so exact
   agreement is not expected. On 12 independent problems (2 to 6 dimensions, 12 to 40
   training points), using a BoTorch GP carrying Optuna's fitted hyperparameters and giving
   both implementations the same max-value samples, they select the same argmax on **10 of
   12** and rank the full 256-candidate set with Spearman correlation averaging 0.92
   (minimum 0.74). The two argmax disagreements are real and unexplained; treat this as
   corroboration rather than proof. Note also that the matched GPs agree on the posterior
   mean to `9.6e-14` but only to `7.2e-03` on the posterior standard deviation.

The posterior max-value sampler was separately checked against 20,000 brute-force joint
posterior draws, matching to within 0.02 standard deviations.

Checks 1 and 2 are the load-bearing ones, since they compare against exact references.
Check 3 compares against a noisy Monte-Carlo estimator of a different quantity.

## Others

### Reference

Zi Wang and Stefanie Jegelka. Max-value Entropy Search for Efficient Bayesian Optimization.
In *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70:3627-3635,
2017\.

### Bibtex

```
@InProceedings{pmlr-v70-wang17e,
  title = {Max-value Entropy Search for Efficient {B}ayesian Optimization},
  author = {Zi Wang and Stefanie Jegelka},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  pages = {3627--3635},
  year = {2017},
  volume = {70},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```
