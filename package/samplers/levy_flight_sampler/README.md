---
author: Aditya Chopra
title: Lévy Flight Sampler
description: A sampler based on Lévy-flight random walks. Lévy flights naturally balance local exploitation with occasional large jumps that escape local optima, mimicking the foraging strategy observed in many animals.
tags: [sampler, levy flight, random walk, nature-inspired, exploration]
optuna_versions: [3.6.0, 4.0.0, 4.8.0]
license: MIT License
---

## Class or Function Names

- `LevyFlightSampler`

## Overview

This sampler proposes new hyperparameter candidates by taking a **Lévy-flight
step** from the current best trial.  A Lévy flight is a random walk whose step
lengths follow a heavy-tailed (Lévy stable) distribution: most steps are
small, enabling fine local search, but occasional large steps jump far away,
helping to escape shallow local optima.

The step is computed via the **Mantegna algorithm**, an efficient
closed-form approximation of the Lévy stable distribution that requires only
standard Gaussian random variates.

### When to use

| Situation                            | Recommendation                            |
| ------------------------------------ | ----------------------------------------- |
| Unimodal or smooth landscapes        | ✅ Converges faster than random search    |
| Multi-modal with many shallow optima | ✅ Large Lévy jumps escape local optima   |
| Highly discontinuous / black-box     | ⚠️ Pair with a large `n_trials` budget    |
| Categorical-heavy search spaces      | ⚠️ Categorical params fall back to random |

### Key parameters

| Parameter    | Default | Effect                                                                                 |
| ------------ | ------- | -------------------------------------------------------------------------------------- |
| `beta`       | `1.5`   | Tail heaviness of the Lévy distribution.  Range `(0, 2]`; `2` ≈ Gaussian, `1` ≈ Cauchy |
| `step_scale` | `0.1`   | Step size as a fraction of each parameter's range                                      |
| `seed`       | `None`  | RNG seed for reproducibility                                                           |

## Installation

```bash
pip install optunahub numpy
```

Or install dependencies directly from the registry:

```bash
pip install -r https://hub.optuna.org/samplers/levy_flight_sampler/requirements.txt
```

## Example

```python
import optuna
import optunahub


def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return (x - 1.5) ** 2 + (y + 2.0) ** 2


mod = optunahub.load_module("samplers/levy_flight_sampler")
sampler = mod.LevyFlightSampler(beta=1.5, step_scale=0.1, seed=42)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)

print(study.best_params)
print(study.best_value)
```

See [`example.py`](https://github.com/optuna/optunahub-registry/blob/main/package/samplers/levy_flight_sampler/example.py) for a full comparison against TPESampler on the Rosenbrock benchmark.

## Algorithm Details

1. **Initialization** — first two trials use random sampling to establish an initial best.
1. **Lévy step** — for each parameter `p` with range `[low, high]`:
   - Draw `u ~ N(0, σ²)` and `v ~ N(0, 1)` using the Mantegna sigma for `beta`.
   - Step length = `step_scale × (high − low) × u / |v|^(1/beta)`.
   - Clip new value to `[low, high]`.
1. **Fallback** — `CategoricalDistribution` and unknown distributions use `RandomSampler`.

The Mantegna sigma is:

```
σ = ( Γ(1+β) · sin(πβ/2) / (Γ((1+β)/2) · β · 2^((β−1)/2)) )^(1/β)
```

## References

- Yang, X.-S. & Deb, S. (2010). *Engineering Optimisation by Cuckoo Search*.
  International Journal of Mathematical Modelling and Numerical Optimisation, 1(4), 330–343.
  https://doi.org/10.1504/IJMMNO.2010.035430

- Mantegna, R. N. (1994). *Fast, accurate algorithm for numerical simulation of Lévy stable stochastic processes*.
  Physical Review E, 49(5), 4677.
  https://doi.org/10.1103/PhysRevE.49.4677
