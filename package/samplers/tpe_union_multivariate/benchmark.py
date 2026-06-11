from __future__ import annotations

import copy
import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import optuna
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.trial import Trial
from tqdm.contrib.concurrent import process_map

from package.samplers.tpe_union_multivariate.sampler import TPEUnionMultivariateSampler


def objective(trial: Trial) -> float:
    x = trial.suggest_categorical("x", [True, False])
    y = trial.suggest_float("y", -1, 1)
    if x is True:
        n = trial.suggest_categorical("n", [True, False])
        if n is True:
            a = trial.suggest_float("a", -1, 1)
            return (a - y) ** 2 + (a + 0.75) ** 2 + 0.025
        b = trial.suggest_float("b", -1, 1)
        return (b - y) ** 2 + (b + 0.25) ** 2 + 0.05
    else:
        m = trial.suggest_categorical("m", [True, False])
        if m is True:
            c = trial.suggest_float("c", -1, 1)
            return (c - y) ** 2 + (c - 0.25) ** 2 + 0.4
        d = trial.suggest_float("d", -1, 1)
        return (d - y) ** 2 + (d - 0.75) ** 2 + 0.01


def run_optimization(n_trials: int, sampler: BaseSampler) -> NDArray[np.float64]:
    sampler_ = copy.deepcopy(sampler)
    study = optuna.create_study(sampler=sampler_, direction="minimize")
    study.enqueue_trial({"x": True, "n": True})
    study.enqueue_trial({"x": True, "n": False})
    study.enqueue_trial({"x": False, "m": True})
    study.enqueue_trial({"x": False, "m": False})

    study.optimize(objective, n_trials=n_trials)
    values = [trial.value for trial in study.trials if trial.value is not None]
    best_values = np.minimum.accumulate(values)
    return best_values


def run_single_benchmark(
    n_trials: int, n_repetitions: int, sampler: BaseSampler, max_workers: int
) -> NDArray[np.float64]:
    results = process_map(
        functools.partial(run_optimization, sampler=sampler),
        (n_trials for _ in range(n_repetitions)),
        max_workers=max_workers,
        desc=sampler.__class__.__name__,
        total=n_repetitions,
        leave=False,
    )
    geometric_mean: NDArray[np.float64] = np.exp(np.sum(np.log(results), axis=0) / n_repetitions)
    return geometric_mean


def run_benchmarks(n_trials: int, n_repetitions: int, max_workers: int) -> None:
    print("Starting benchmarks...")
    r1 = run_single_benchmark(n_trials, n_repetitions, RandomSampler(seed=42), max_workers)
    r2 = run_single_benchmark(
        n_trials,
        n_repetitions,
        TPEUnionMultivariateSampler(n_startup_trials=8, seed=42),
        max_workers,
    )

    plt.plot(r1, label="Random Search")
    plt.plot(r2, label="TPE Union Multivariate Sampler", linewidth=2, linestyle="--")

    plt.legend()
    plt.xlabel("Trials")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Conditional Search Space Convergence Comparison")
    plt.savefig("benchmark.png", dpi=400)
    print("Benchmark complete! Saved results plot to benchmark.png")


if __name__ == "__main__":
    np.random.seed(42)
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=optuna.exceptions.ExperimentalWarning)
    run_benchmarks(n_trials=150, n_repetitions=32, max_workers=4)
