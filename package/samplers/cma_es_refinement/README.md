---
author: Elias Munk
title: CMA-ES with Multi-Stage Refinement Sampler
description: Three-phase sampler combining Sobol QMC initialization, CMA-ES optimization, and multi-stage Gaussian refinement. Achieves 25% lower regret than pure CMA-ES on BBOB benchmarks by using remaining trial budget for targeted local search.
tags: [sampler, CMA-ES, local search, refinement, BBOB, black-box optimization]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

CMA-ES is the gold standard for continuous black-box optimization, but it has diminishing returns: after convergence, additional CMA-ES trials provide little improvement. This sampler addresses that by splitting the trial budget into three phases:

1. **Sobol QMC** (8 trials) — quasi-random space-filling initialization
2. **CMA-ES** (132 trials) — covariance matrix adaptation for main optimization
3. **Multi-stage Gaussian refinement** (60 trials) — targeted local search around the best point with decreasing perturbation scale

The refinement phase exploits a key property of Optuna studies: `study.best_value` tracks the global best across all trials. Any improvement from perturbation is kept, while failed perturbations don't hurt the result.

### BBOB Benchmark Results

Evaluated on all 24 BBOB functions (5D, 10 seeds, 200 trials each):

| Sampler | Mean Normalized Regret | vs Random |
|---------|----------------------|-----------|
| Random baseline | 1.0000 | — |
| Default TPE | 0.2463 | 75% better |
| CMA-ES (tuned) | 0.2004 | 80% better |
| **CMA-ES + Refinement** | **0.1501** | **85% better** |

Per-category breakdown:

| Category | CMA-ES | CMA-ES + Refinement | Improvement |
|----------|--------|---------------------|-------------|
| Separable | 0.1682 | 0.1161 | -31% |
| Low conditioning | 0.0281 | 0.0311 | +11% |
| High conditioning | 0.0592 | 0.0511 | -14% |
| Multimodal (global) | 0.2508 | 0.1663 | -34% |
| Multimodal (weak) | 0.4615 | 0.3623 | -21% |

The improvement is especially strong on multimodal functions, where the refinement phase finds better local optima that CMA-ES misses after convergence.

### Benchmark: BBOB (Black-Box Optimization Benchmarking)

The [BBOB benchmark suite](https://numbbo.github.io/coco/testsuites/bbob) is the gold standard for evaluating continuous black-box optimizers, used in GECCO competitions worldwide. It consists of 24 noiseless functions spanning 5 difficulty categories:

- **Separable** (f1–f5): Functions where dimensions are independent
- **Low conditioning** (f6–f9): Well-conditioned functions with moderate difficulty
- **High conditioning** (f10–f14): Ill-conditioned functions requiring covariance adaptation
- **Multimodal global** (f15–f19): Multiple global optima with visible structure
- **Multimodal weak** (f20–f24): Deceptive multimodal landscapes (hardest category)

**Evaluation protocol:**
- Dimension: 5
- Trials per run: 200
- Seeds: 10 (for statistical robustness)
- Metric: Normalized regret = `(sampler_best - f_opt) / (random_best - f_opt)`
  - 0.0 = optimal, 1.0 = random-level performance
- Final score: Mean normalized regret across all 24 functions and 10 seeds

The optimal values (`f_opt`) were computed via `scipy.differential_evolution` with 5 restarts. Random baselines were computed from 10 seeds of 200 random trials each.

Results were validated across 97 experiments with deterministic reproducibility (same seed always produces the same regret). Full experiment logs available at [github.com/EliMunkey/autoresearch-optuna](https://github.com/EliMunkey/autoresearch-optuna).

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
    medium_frac: float = 0.5,
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
- **`medium_frac`** — Fraction of refinement phase allocated to medium perturbation. Default: `0.5`.
- **`seed`** — Random seed for reproducibility. Default: `None`.

## Installation

No additional packages beyond `optuna` and `optunahub` are required.

## Example

```python
import optuna
import optunahub

module = optunahub.load_module("samplers/cma_es_refinement")
sampler = module.CmaEsRefinementSampler(seed=42)

study = optuna.create_study()
study.optimize(objective, n_trials=200)
```

For a complete working example, see [example.py](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/cma_es_refinement/example.py).
