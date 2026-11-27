---
author: Ahmet Emin Karkinli
title: Multi-population based Differential Evolution (MDE) Sampler
description: A parameter-free differential evolution variant that evolves a randomly drawn sub-population of a three times larger full population in every generation, with a self-adaptive scale factor and crossover map.
tags: [sampler, differential evolution, evolutionary algorithm, parameter-free, metaheuristics]
optuna_versions: [5.1.0]
license: MIT License
---

## Abstract

Multi-population based Differential Evolution (MDE) is a differential evolution variant that has no control parameters to tune apart from the population size. It keeps a full population `A` of `3N` vectors. In every generation it draws a sub-population `B` of `N` random members of `A` and builds one trial vector per member:

- the scale factor is drawn anew for every vector as `|a - rand^b| * randn^d` with random integer exponents, which mixes many small and occasional large steps,
- the difference vector `dx - dy` mixes the best solution, two random members of `B` and the current member coordinate-wise,
- a random, parameter-free crossover map decides which coordinates are taken from the mutant,
- a tiny multiplicative noise term, computed from the previous sub-population, is added,
- variables that leave the box are re-sampled uniformly inside the bounds.

Trial vectors that are better than their parents replace them (greedy selection), the sub-population is written back into `A`, and the best member of `A` guides the next generation. The algorithm was proposed for detecting object boundaries in point clouds (Karkinli, 2023).

This package is a faithful port of the original MATLAB implementation; out-of-bounds variables are handled as in Eq. 15 of the paper (uniform re-sampling inside the bounds). With `search_space` given, a fixed seed and sequential optimization, the sampler proposes exactly the same points as a stand-alone Python port of the MATLAB code that was checked against MATLAB.

## APIs

- `MDESampler(search_space: dict[str, BaseDistribution] | None = None, population_size: int = 30, seed: int | None = None)`
  - `search_space`: A dictionary containing the search space that defines the parameter space. The keys are the parameter names and the values are [the parameter's distribution](https://optuna.readthedocs.io/en/stable/reference/distributions.html). If the search space is not provided, the sampler infers it from completed trials (intersection search space); in that case the very first trial is sampled randomly, before the MDE population is initialized.
    Example:
    ```python
    search_space = {
        "x": optuna.distributions.FloatDistribution(-10, 10),
        "y": optuna.distributions.FloatDistribution(-10, 10),
    }
    MDESampler(search_space=search_space)
    ```
  - `population_size`: `N`, the number of trial vectors evaluated in every generation (must be at least 3). The full population has `3 * N` members, so the first `3 * N` trials evaluate the random initial population and each later generation consists of `N` trials.
  - `seed`: Seed for the random number generators.

### How generations map to trials

- Generation 0: `3 * population_size` trials evaluate the uniformly initialized full population.
- Generation `g >= 1`: `population_size` trials, one for each trial vector of the sub-population.
- The population update (greedy selection, write-back, best solution and noise) runs as soon as every trial of the current generation has finished, and the next generation is created right away.

### Supported parameters and trial states

- Float parameters are optimized directly.
- Integer parameters and float parameters with `step` are optimized in a continuous relaxation and rounded to the nearest valid value. Log-scale parameters are optimized in log space.
- Categorical parameters, and parameters that are not in the relative search space, are sampled by `optuna.samplers.RandomSampler`.
- Pruned and failed trials are treated as having the objective value `+inf`, so they never replace a population member.
- `direction="maximize"` is supported (the objective is negated internally). Multi-objective optimization is not supported.

### Parallel optimization

The sampler can be used with `n_jobs > 1`. While the last trials of a generation are still running, there is no trial vector left to hand out; such trials are sampled by `RandomSampler` and are not used by MDE. In a test with `population_size=30`, `n_jobs=4` and 3000 trials, about 5% of the trials were sampled this way. Results with `n_jobs > 1` are not reproducible, as for other Optuna samplers.

### Limitations

- The population state lives in the sampler object. Resuming a study with a new sampler instance starts from a new random population.
- If the search space (or the study) changes, the sampler starts again with a new random population.
- A trial that is asked but never told blocks the current generation; later trials are then sampled randomly.

## Example

```python
import numpy as np
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    # 10-dimensional Rastrigin function (global minimum 0 at the origin).
    x = np.array([trial.suggest_float(f"x{i}", -5.12, 5.12) for i in range(10)])
    return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


module = optunahub.load_module(package="samplers/mde")
sampler = module.MDESampler(population_size=30, seed=42)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=3000)
print(study.best_value, study.best_params)
```

## Others

### Benchmark

Best objective value after 3000 trials in 10 dimensions (`population_size=30`, box `[-5.12, 5.12]^10`), median (min, max) over seeds 0 to 9:

| Function  | MDESampler          | RandomSampler       |
| --------- | ------------------- | ------------------- |
| Sphere    | 0.153 (0.069–0.312) | 17.38 (13.85–20.94) |
| Rastrigin | 10.89 (9.44–17.75)  | 78.32 (61.10–86.36) |

The MDESampler values are identical to the stand-alone reference implementation run with the same seeds and the same number of function evaluations.

### Reference

Karkinli, A.E. (2023). Detection of object boundary from point cloud by using multi-population based differential evolution algorithm. Neural Computing and Applications, 35, 5193–5206. https://doi.org/10.1007/s00521-022-07969-w

Original MATLAB implementation: https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm

### Bibtex

```
@article{karkinli2023mde,
    title={Detection of object boundary from point cloud by using multi-population based differential evolution algorithm},
    author={Karkinli, Ahmet Emin},
    journal={Neural Computing and Applications},
    volume={35},
    pages={5193--5206},
    year={2023},
    doi={10.1007/s00521-022-07969-w}
}
```
