---
author: Elias Munk
title: CMA-ES with Quasi-Random Refinement Sampler
description: Three-phase sampler combining Sobol QMC initialization, CMA-ES optimization, and quasi-random Gaussian refinement using Sobol-based perturbation vectors. Achieves 36% lower regret than pure CMA-ES on BBOB benchmarks.
tags: [sampler, CMA-ES, local search, refinement, BBOB, black-box optimization, quasi-random]
optuna_versions: [4.7.0]
license: MIT License
---

## Abstract

CMA-ES is the gold standard for continuous black-box optimization, but it has diminishing returns: after convergence, additional CMA-ES trials provide little improvement. This sampler addresses that by splitting the trial budget into three phases:

1. **Sobol QMC** (8 trials) — quasi-random space-filling initialization
1. **CMA-ES** (132 trials) — covariance matrix adaptation for main optimization
1. **Quasi-random Gaussian refinement** (60 trials) — targeted local search around the best point using Sobol-based perturbation vectors with exponentially decaying scale

The refinement phase uses quasi-random Sobol sequences transformed via inverse CDF to generate Gaussian-distributed perturbation vectors. Compared to pseudo-random Gaussian perturbation, this provides more uniform directional coverage in high-dimensional spaces — systematically exploring directions that pseudo-random sampling might miss.

The perturbation scale follows an exponential decay: `sigma(n) = 0.13 * exp(-0.11 * n)`, starting wide for basin exploration and tightening for precise convergence.

## Benchmark Results

Evaluated on the [BBOB benchmark suite](https://numbbo.github.io/coco/testsuites/bbob) — 24 noiseless black-box optimization functions spanning 5 difficulty categories, used as the gold standard in GECCO competitions. All results use 5 dimensions, 10 random seeds, and 200 trials per run.

**Metric:** Normalized regret = `(sampler_best - f_opt) / (random_best - f_opt)` where 0.0 = optimal and 1.0 = random-level. Optimal values computed via `scipy.differential_evolution` (5 restarts). Random baselines from 10 seeds of 200 random trials.

| Sampler                 | Mean Normalized Regret | vs Random      |
| ----------------------- | ---------------------- | -------------- |
| Random baseline         | 1.0000                 | —              |
| Default TPE             | 0.2463                 | 75% better     |
| CMA-ES (tuned)          | 0.2004                 | 80% better     |
| **CMA-ES + Refinement** | **0.1284**             | **87% better** |

Per-category breakdown:

| Category            | Functions | CMA-ES | CMA-ES + Refinement | Change |
| ------------------- | --------- | ------ | ------------------- | ------ |
| Separable           | f1–f5     | 0.1682 | 0.0996              | -41%   |
| Low conditioning    | f6–f9     | 0.0281 | 0.0244              | -13%   |
| High conditioning   | f10–f14   | 0.0592 | 0.0513              | -13%   |
| Multimodal (global) | f15–f19   | 0.2508 | 0.1374              | -45%   |
| Multimodal (weak)   | f20–f24   | 0.4615 | 0.3084              | -33%   |

The improvement is strongest on separable and multimodal functions, where the quasi-random refinement's uniform directional coverage systematically finds improvements that pseudo-random perturbation misses. Results are deterministic and reproducible. Full experiment logs (135 experiments): [github.com/EliMunkey/autoresearch-optuna](https://github.com/EliMunkey/autoresearch-optuna).

### Cross-validation on standard test functions

Independent validation on 8 standard test functions (Sphere, Rosenbrock, Rastrigin, Ackley, Griewank, Levy, Styblinski-Tang, Schwefel) with 5 seeds, 200 trials. Includes TPE with `multivariate=True` as a stronger baseline.

**5D results:**

| Sampler                 | Mean Normalized Regret | vs Random      |
| ----------------------- | ---------------------- | -------------- |
| Random baseline         | 1.0000                 | —              |
| TPE                     | 0.2437                 | 76% better     |
| TPE (multivariate)      | 0.3365                 | 66% better     |
| CMA-ES                  | 0.2715                 | 73% better     |
| **CMA-ES + Refinement** | **0.2038**             | **80% better** |

**10D results — the advantage widens at higher dimensions:**

| Sampler                 | Mean Normalized Regret | vs Random      |
| ----------------------- | ---------------------- | -------------- |
| Random baseline         | 1.0000                 | —              |
| TPE                     | 0.4803                 | 52% better     |
| TPE (multivariate)      | 0.5463                 | 45% better     |
| CMA-ES                  | 0.3737                 | 63% better     |
| **CMA-ES + Refinement** | **0.1719**             | **83% better** |

Per-function breakdown (10D):

| Function        | Category   | TPE    | TPE(mv) | CMA-ES | CMA-ES+Refine |
| --------------- | ---------- | ------ | ------- | ------ | ------------- |
| Sphere          | Unimodal   | 0.2751 | 0.2699  | 0.0312 | 0.0000        |
| Rosenbrock      | Unimodal   | 0.1139 | 0.0851  | 0.0071 | 0.0059        |
| Rastrigin       | Multimodal | 0.6891 | 0.8187  | 0.6566 | 0.0000        |
| Ackley          | Multimodal | 0.6146 | 0.7282  | 0.3704 | 0.0000        |
| Griewank        | Multimodal | 0.3050 | 0.2782  | 0.0444 | 0.0000        |
| Levy            | Multimodal | 0.5383 | 0.3941  | 0.0965 | 0.0188        |
| Styblinski-Tang | Multimodal | 0.5651 | 0.8238  | 0.6355 | 0.3981        |
| Schwefel        | Deceptive  | 0.7413 | 0.9723  | 1.1481 | 0.9526        |

**Limitation:** On deceptive functions like Schwefel — where the global optimum is far from typical local optima — the refinement phase can reinforce a suboptimal basin. CMA-ES itself struggles on Schwefel (regret >1.0), and refinement does not recover from that. For problems with known deceptive structure, consider using TPE or increasing the CMA-ES budget.

## APIs

### `CmaEsRefinementSampler`

```python
CmaEsRefinementSampler(
    *,
    n_startup_trials: int = 8,
    cma_n_trials: int = 132,
    popsize: int = 6,
    sigma0: float = 0.2,
    sigma_start: float = 0.13,
    decay_rate: float = 0.11,
    seed: int | None = None,
)
```

#### Parameters

- **`n_startup_trials`** — Number of Sobol QMC initialization trials. Powers of 2 recommended. Default: `8`.
- **`cma_n_trials`** — Number of CMA-ES optimization trials. Default: `132`.
- **`popsize`** — CMA-ES population size. Default: `6`.
- **`sigma0`** — CMA-ES initial step size. Default: `0.2`.
- **`sigma_start`** — Initial refinement perturbation scale as a fraction of parameter range. Default: `0.13`.
- **`decay_rate`** — Exponential decay rate for refinement perturbation scale. Default: `0.11`.
- **`seed`** — Random seed for reproducibility. Default: `None`.

### `CmaEsRefinementSampler.for_budget`

```python
CmaEsRefinementSampler.for_budget(n_trials, *, seed=None, **kwargs)
```

Factory method that scales phase boundaries proportionally to the trial budget.
The default parameters are tuned for 200 trials; use this when running a different number.

```python
# 1000-trial study with auto-scaled phases
sampler = module.CmaEsRefinementSampler.for_budget(1000, seed=42)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=1000)
```

## Installation

```shell
$ pip install optunahub cmaes scipy
```

## Example

```python
import math
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    n = 5
    variables = [trial.suggest_float(f"x{i}", -5.12, 5.12) for i in range(n)]
    return 10 * n + sum(x**2 - 10 * math.cos(2 * math.pi * x) for x in variables)


module = optunahub.load_module("samplers/cma_es_refinement")
sampler = module.CmaEsRefinementSampler(seed=42)

study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=200)

print(f"Best value: {study.best_value:.6f}")
print(f"Best params: {study.best_params}")
```
