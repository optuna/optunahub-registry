---
author: Maxim Kirby
title: 'ORTHOBO: Orthogonal Bayesian Hyperparameter Optimization'
description: A drop-in Optuna sampler that reduces acquisition estimation noise in marginal Bayesian optimization using orthogonal score-function control variates, improving candidate ranking stability and optimization performance.
tags: [sampler, bayesian-optimization, gaussian-process, variance-reduction, botorch]
optuna_versions: [4.8.0]
license: MIT License
---

## Abstract

Standard Bayesian optimization handles uncertainty by taking random samples of model parameters, but this random sampling creates noise. This noise can cause the optimizer to accidentally rank bad configurations above good ones and steer the search in the wrong direction. This issue is especially noticeable in complex tasks with many variables or when computational limits restrict the number of samples you can take.

This package implements OrthoBO, a framework introduced in [this paper from May 2026](https://arxiv.org/abs/2605.06454). OrthoBO uses an orthogonal correction to cancel out the noise caused by finite sampling. It keeps the estimates accurate while making the candidate rankings much more stable. The result is better optimization performance.

OrthoBO is designed as a drop-in replacement for Optuna's default BoTorch sampler. It works with almost any standard Bayesian optimization problem and requires absolutely no changes to your objective function or search space.

## APIs

### `OrthoBoSampler`

```python
OrthoBoSampler(
    n_startup_trials: int = 10,
    mc_budget: int = 64,
    use_orthogonal_correction bool = True,
    seed: int | None = None,
)
```

**Parameters:**

- `n_startup_trials`:
  Number of initial quasi-random (Sobol) trials before BO begins. Defaults to 10.
- `mc_budget`:
  The number of GP models sampled. Higher values reduce variance but increase compute time. Defaults to 64.
- `use_orthogonal_correction`:
  If True, applies the orthogonal score-function control variate for variance reduction (OrthoBO mode). If False, falls back to plain MC marginalisation over the hyperposterior (Naive Marginal BO mode). Defaults to True
- `seed`:
  Random seed for the Sobol startup sampler. Set for reproducibility. Defaults to None.

## Installation

```bash
pip install torch botorch gpytorch optuna optuna-integration
```

## Example

```python
import optuna
import optunahub

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    return (x - 2) ** 2 + (y + 1) ** 2

OrthoBoSampler = optunahub.load_module("samplers/orthobo").OrthoBoSampler

sampler = OrthoBoSampler(n_startup_trials=10, mc_budget=64, use_orthogonal_correction=True)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=50)

print(f"Best value: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

## Benchmark Results

Mean best-so-far regret ± 1 std across 5 random seeds. OrthoBO (green) is
compared against Vanilla BO (standard single-GP qLogEI, blue) and Naive
Marginal BO (MC marginalisation without orthogonal correction, orange).

### Hartmann-6 (6 dimensions)

![Hartmann6](figures/Hartmann6.png)

### Ackley-10 (10 dimensions)

![Ackley10](figures/Ackley10.png)

### Levy-16 (16 dimensions)

![Levy16](figures/Levy16.png)

OrthoBO's advantage is most pronounced on higher-dimensional problems where acquisition estimation noise has a larger effect on candidate rankings. On lower-dimensional problems (Hartmann-6) where the GP fits well with few observations, the methods converge to similar performance.

## Reference

```bibtex
@misc{schröder2026orthoboorthogonalbayesianhyperparameter,
      title={ORTHOBO: Orthogonal Bayesian Hyperparameter Optimization}, 
      author={Maresa Schröder and Pascal Janetzky and Michael Klar and Stefan Feuerriegel},
      year={2026},
      eprint={2605.06454},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.06454}, 
}
```
