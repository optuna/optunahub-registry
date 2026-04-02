---
author: Yasunori Morishima
title: GP-Based Samplers with Alternative Acquisition Functions
description: GP-based Bayesian optimization samplers with Probability of Improvement (PI), Upper Confidence Bound (UCB), and Thompson Sampling (TS) acquisition functions.
tags: [sampler, Bayesian optimization, Gaussian process, acquisition function]
optuna_versions: [4.8.0]
license: MIT License
---

## Class or Function Names

- GPEISampler (alias for `optuna.samplers.GPSampler`)
- GPPISampler
- GPUCBSampler
- GPTSSampler

## Installation

```bash
pip install scipy torch
```

## Overview

Optuna's built-in `GPSampler` only supports Expected Improvement (EI) as an acquisition function. This package extends `GPSampler` with three additional acquisition functions commonly used in Bayesian optimization:

| Sampler        | Acquisition Function            | Description                                                                            |
| -------------- | ------------------------------- | -------------------------------------------------------------------------------------- |
| `GPEISampler`  | Expected Improvement (EI)       | Alias for `optuna.samplers.GPSampler`. Balances improvement magnitude and probability. |
| `GPPISampler`  | Probability of Improvement (PI) | Selects the point most likely to improve over the current best. More exploitative.     |
| `GPUCBSampler` | Upper Confidence Bound (UCB)    | Balances exploration and exploitation via a `beta` parameter.                          |
| `GPTSSampler`  | Thompson Sampling (TS)          | Samples from the GP posterior and maximizes the sample. No tuning parameter needed.    |

All samplers inherit from `GPSampler` and reuse its GP fitting, search-space handling, and acquisition function optimization machinery. For multi-objective or constrained optimization, they automatically fall back to the parent `GPSampler` behaviour.

### When to use which?

- **EI** (default): Good general-purpose choice. Well-understood theoretical properties.
- **PI**: When you want to quickly find *any* improvement. Can get stuck in local optima.
- **UCB**: When you want explicit control over exploration vs. exploitation (`beta` parameter).
- **TS**: When you want automatic exploration-exploitation balance without tuning. Good for parallel optimization.

## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return -(x**2 + y**2)


mod = optunahub.load_module("samplers/gp_acqf_samplers")

# Probability of Improvement
study = optuna.create_study(direction="maximize", sampler=mod.GPPISampler(seed=42))
study.optimize(objective, n_trials=50)

# Upper Confidence Bound (beta controls exploration)
study = optuna.create_study(direction="maximize", sampler=mod.GPUCBSampler(beta=2.0, seed=42))
study.optimize(objective, n_trials=50)

# Thompson Sampling
study = optuna.create_study(direction="maximize", sampler=mod.GPTSSampler(seed=42))
study.optimize(objective, n_trials=50)
```

## API Reference

### GPPISampler

Same arguments as `optuna.samplers.GPSampler`.

### GPUCBSampler

| Argument | Type    | Default | Description                                                                   |
| -------- | ------- | ------- | ----------------------------------------------------------------------------- |
| `beta`   | `float` | `2.0`   | Exploration-exploitation trade-off. Larger values encourage more exploration. |

Plus all arguments from `optuna.samplers.GPSampler`.

### GPTSSampler

| Argument         | Type  | Default | Description                                                    |
| ---------------- | ----- | ------- | -------------------------------------------------------------- |
| `n_rff_features` | `int` | `512`   | Number of random Fourier features for posterior approximation. |

Plus all arguments from `optuna.samplers.GPSampler`.

## References

- Srinivas et al., "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design" (2010) — UCB theory.
- Rahimi & Recht, "Random Features for Large-Scale Kernel Machines" (2007) — RFF for Thompson Sampling.
- Garnett, "Bayesian Optimization" (2023) — [Textbook](https://bayesoptbook.com/book/bayesoptbook.pdf) referenced in the issue.
