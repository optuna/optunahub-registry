---
author: Matthew Busel
title: Steady-State Genetic Algorithm Sampler (strategy-genome)
description: A steady-state genetic algorithm sampler with SBX crossover, polynomial mutation and tournament selection from an elite pool, with NSGA-II style ranking for multi-objective studies.
tags: [sampler, genetic algorithm, evolutionary algorithm, multi-objective optimization, NSGA-II]
optuna_versions: [4.6.0, 5.0.0]
license: MIT License
---

## Abstract

This package provides `GeneticStrategySampler`, a steady-state genetic algorithm sampler from the [strategy-genome](https://github.com/Mattbusel/strategy-genome) library.
After a random warm-up of `population_size` trials, each new trial is the child of two parents picked by tournament from the best `population_size` completed trials (the elite pool).
Numeric parameters are recombined with simulated binary crossover (SBX), categorical parameters with uniform crossover, and the child is then perturbed with polynomial mutation.
Multi-objective studies rank the elite pool with non-dominated sorting and crowding distance, in the style of NSGA-II.

Because there is no generation barrier, every child is bred from whatever trials have finished so far.
This makes the sampler a good fit for parallel workers and batched `study.ask()` loops, where a generational GA such as `NSGAIISampler` has to wait for a full generation.
It handles float, int (including log and stepped domains) and categorical parameters in one search space.

## APIs

- `GeneticStrategySampler(*, population_size: int = 50, crossover_prob: float = 0.9, mutation_prob: float | None = None, sbx_eta: float = 15.0, mutation_eta: float = 20.0, tournament_size: int = 3, seed: int | None = None, independent_sampler: BaseSampler | None = None)`
  - `population_size`: Number of random warm-up trials, and size of the elite pool that parents are drawn from. Must be at least 2.
  - `crossover_prob`: Probability that a child is produced by crossover. Otherwise the child is a mutated copy of one parent.
  - `mutation_prob`: Per-parameter mutation probability. Defaults to `1 / n_params`.
  - `sbx_eta`: SBX distribution index. Higher values keep children closer to their parents.
  - `mutation_eta`: Polynomial mutation distribution index. Higher values give smaller mutations.
  - `tournament_size`: Each parent is the best of this many randomly chosen elite pool members.
  - `seed`: Seed for the random number generator. Sampling is reproducible for a sequential study.
  - `independent_sampler`: Sampler used during the warm-up and for parameters outside the shared search space. Defaults to `optuna.samplers.RandomSampler(seed=seed)`.

The sampler works on the intersection search space of completed trials.
Parameters that appear only in some trials (conditional parameters) are sampled by `independent_sampler`.

## Installation

The sampler is implemented in the `strategy-genome` package on PyPI.

```shell
$ pip install "strategy-genome[optuna]"
```

The same dependency is listed in this package's `requirements.txt`:

```shell
$ pip install -r https://hub.optuna.org/samplers/strategy_genome/requirements.txt
```

## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_int("y", -10, 10)
    kind = trial.suggest_categorical("kind", ["a", "b"])
    return (x - 3) ** 2 + (y + 2) ** 2 + (0 if kind == "b" else 5)


module = optunahub.load_module(package="samplers/strategy_genome")
sampler = module.GeneticStrategySampler(population_size=30, seed=1)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=300)
print(study.best_params, study.best_value)
```

Multi-objective studies need no extra configuration:

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)
    return 4 * x**2 + 4 * y**2, (x - 5) ** 2 + (y - 5) ** 2


module = optunahub.load_module(package="samplers/strategy_genome")
sampler = module.GeneticStrategySampler(population_size=20, seed=0)
study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
study.optimize(objective, n_trials=200)
print(len(study.best_trials))
```

## Others

### Source and maintenance

The sampler code lives in the `strategy_genome.optuna` module of [strategy-genome](https://github.com/Mattbusel/strategy-genome) and is released on [PyPI](https://pypi.org/project/strategy-genome/).
This OptunaHub package re-exports it, so fixes ship through normal PyPI releases.
Bug reports are welcome on the [strategy-genome issue tracker](https://github.com/Mattbusel/strategy-genome/issues).
