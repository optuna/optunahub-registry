---
author: Elias Munk
title: CMA-ES with Multi-Stage Refinement Sampler
description: Three-phase sampler combining Sobol QMC initialization, CMA-ES optimization, and multi-stage Gaussian refinement. Achieves 25% lower regret than pure CMA-ES on BBOB benchmarks by using remaining trial budget for targeted local search.
tags: [sampler, CMA-ES, local search, refinement, BBOB, black-box optimization]
optuna_versions: [4.7.0]
license: MIT License
---

## Abstract

CMA-ES is the gold standard for continuous black-box optimization, but it has diminishing returns: after convergence, additional CMA-ES trials provide little improvement. This sampler addresses that by splitting the trial budget into three phases:

1. **Sobol QMC** (8 trials) — quasi-random space-filling initialization
2. **CMA-ES** (132 trials) — covariance matrix adaptation for main optimization
3. **Multi-stage Gaussian refinement** (60 trials) — targeted local search around the best point with decreasing perturbation scale

The refinement phase exploits a key property of Optuna studies: `study.best_value` tracks the global best across all trials. Any improvement from perturbation is kept, while failed perturbations don't hurt the result.

## Benchmark Results

Evaluated on the [BBOB benchmark suite](https://numbbo.github.io/coco/testsuites/bbob) — 24 noiseless black-box optimization functions spanning 5 difficulty categories, used as the gold standard in GECCO competitions. All results use 5 dimensions, 10 random seeds, and 200 trials per run.

**Metric:** Normalized regret = `(sampler_best - f_opt) / (random_best - f_opt)` where 0.0 = optimal and 1.0 = random-level. Optimal values computed via `scipy.differential_evolution` (5 restarts). Random baselines from 10 seeds of 200 random trials.

| Sampler | Mean Normalized Regret | vs Random |
|---------|----------------------|-----------|
| Random baseline | 1.0000 | — |
| Default TPE | 0.2463 | 75% better |
| CMA-ES (tuned) | 0.2004 | 80% better |
| **CMA-ES + Refinement** | **0.1501** | **85% better** |

Per-category breakdown:

| Category | Functions | CMA-ES | CMA-ES + Refinement | Change |
|----------|-----------|--------|---------------------|--------|
| Separable | f1–f5 | 0.1682 | 0.1161 | -31% |
| Low conditioning | f6–f9 | 0.0281 | 0.0311 | +11% |
| High conditioning | f10–f14 | 0.0592 | 0.0511 | -14% |
| Multimodal (global) | f15–f19 | 0.2508 | 0.1663 | -34% |
| Multimodal (weak) | f20–f24 | 0.4615 | 0.3623 | -21% |

The improvement is strongest on multimodal functions, where the refinement phase fine-tunes solutions that CMA-ES leaves on the table after convergence. Results are deterministic and reproducible. Full experiment logs (97 experiments): [github.com/EliMunkey/autoresearch-optuna](https://github.com/EliMunkey/autoresearch-optuna).

## APIs

### `CmaEsRefinementSampler`

```python
CmaEsRefinementSampler(
    *,
    n_startup_trials: int = 8,
    cma_n_trials: int = 132,
    popsize: int = 6,
    sigma0: float = 0.2,
    medium_sigma_frac: float = 0.01,
    tight_sigma_frac: float = 0.002,
    n_medium_refine_trials: int = 30,
    seed: int | None = None,
)
```

#### Parameters

- **`n_startup_trials`** — Number of Sobol QMC initialization trials. Powers of 2 recommended. Default: `8`.
- **`cma_n_trials`** — Number of CMA-ES optimization trials. Default: `132`.
- **`popsize`** — CMA-ES population size. Default: `6`.
- **`sigma0`** — CMA-ES initial step size. Default: `0.2`.
- **`medium_sigma_frac`** — Perturbation scale for medium refinement (fraction of parameter range). Default: `0.01`.
- **`tight_sigma_frac`** — Perturbation scale for tight refinement (fraction of parameter range). Default: `0.002`.
- **`n_medium_refine_trials`** — Number of medium-perturbation trials before switching to tight. Default: `30`.
- **`seed`** — Random seed for reproducibility. Default: `None`.

## Installation

No additional packages beyond `optuna` and `optunahub` are required.

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
