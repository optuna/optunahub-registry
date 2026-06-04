"""Benchmark the HierarchicalTPESampler against Optuna baselines on a conditional objective.

The objective has an always-present ``y`` coupled to whichever conditional leaf the categorical
gates activate, so cross-subspace correlation matters. Each sampler is run for ``N_REPS``
repetitions at a fixed ``n_ei_candidates`` (its best from a prior sweep), and the geometric mean
of the best-so-far loss is plotted against the trial count.

The ``MV-TPE (group=False)`` curve reflects whatever the installed Optuna does for a
``multivariate=True, group=False`` sampler on a dynamic search space: independent fallback on
released Optuna (<= 4.8), or the union-search-space approach when run against the optuna#6697
branch.

Usage:
    python package/samplers/hierarchical_tpe/benchmark.py
"""

from __future__ import annotations

import multiprocessing
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
import optunahub


N_TRIALS = 500
N_REPS = 128
N_STARTUP = 10
N_WORKERS = 6
IMAGE_PATH = "benchmark.png"

HierarchicalTPESampler = optunahub.load_local_module(
    package="samplers/hierarchical_tpe", registry_root="package/"
).HierarchicalTPESampler


def objective(trial: optuna.Trial) -> float:
    """Evaluate the conditional benchmark objective.

    The always-present parameter ``y`` is coupled to whichever leaf parameter is active, so a
    sampler benefits from capturing the correlation between the root group ``{x, y}`` and the
    conditional leaves. Branch offsets are competitive (0.01 - 0.4); the global optimum is in
    the ``d`` branch at a loss of 0.01.

    Args:
        trial: The Optuna trial to evaluate.

    Returns:
        The objective value (loss) to minimize.
    """
    x = trial.suggest_categorical("x", [True, False])
    y = trial.suggest_float("y", -1, 1)
    if x:
        n = trial.suggest_categorical("n", [True, False])
        if n:
            a = trial.suggest_float("a", -1, 1)
            return (a - y) ** 2 + (a + 0.75) ** 2 + 0.025
        b = trial.suggest_float("b", -1, 1)
        return (b - y) ** 2 + (b + 0.25) ** 2 + 0.05
    m = trial.suggest_categorical("m", [True, False])
    if m:
        c = trial.suggest_float("c", -1, 1)
        return (c - y) ** 2 + (c - 0.25) ** 2 + 0.4
    d = trial.suggest_float("d", -1, 1)
    return (d - y) ** 2 + (d - 0.75) ** 2 + 0.01


def conditional_fn(params: dict[str, object]) -> list[str]:
    """Exact map of the benchmark objective's conditional structure.

    Args:
        params: The parameter values chosen so far (external representation).

    Returns:
        The names of the parameters the objective will request next given ``params``.
    """
    if "x" not in params:
        return []
    if params["x"]:
        return ["n"] if "n" not in params else (["a"] if params["n"] else ["b"])
    return ["m"] if "m" not in params else (["c"] if params["m"] else ["d"])


def make_samplers(seed: int) -> dict[str, BaseSampler]:
    """Build the samplers to compare, each at its best ``n_ei_candidates`` from a prior sweep.

    Args:
        seed: The random seed for every sampler.

    Returns:
        A mapping from a human-readable sampler name to a freshly constructed sampler.
    """
    return {
        "Random": RandomSampler(seed=seed),
        "TPE (independent)": TPESampler(seed=seed, n_startup_trials=N_STARTUP, n_ei_candidates=16),
        "MV-TPE (group=False)": TPESampler(
            seed=seed,
            n_startup_trials=N_STARTUP,
            n_ei_candidates=64,
            multivariate=True,
            group=False,
        ),
        "MV-TPE (group=True)": TPESampler(
            seed=seed,
            n_startup_trials=N_STARTUP,
            n_ei_candidates=16,
            multivariate=True,
            group=True,
        ),
        "Hierarchical (learned)": HierarchicalTPESampler(
            seed=seed, n_startup_trials=N_STARTUP, n_ei_candidates=128
        ),
        "Hierarchical (exact map)": HierarchicalTPESampler(
            seed=seed,
            n_startup_trials=N_STARTUP,
            n_ei_candidates=128,
            conditional_fn=conditional_fn,
        ),
    }


def _run_one(rep: int) -> dict[str, np.ndarray]:
    """Run every sampler once for a single repetition.

    Args:
        rep: The repetition index, used as the seed for every sampler.

    Returns:
        A mapping from sampler name to its best-so-far loss curve (an array of length
        ``N_TRIALS``).
    """
    warnings.simplefilter("ignore")
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    result = {}
    for name, sampler in make_samplers(rep).items():
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=N_TRIALS)
        values = np.array([t.value for t in study.trials], dtype=float)
        result[name] = np.minimum.accumulate(values)
    return result


def main() -> None:
    """Run the benchmark in parallel and save the geometric-mean convergence plot."""

    names = list(make_samplers(0).keys())
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        results = pool.map(_run_one, range(N_REPS))

    trials = np.arange(1, N_TRIALS + 1)
    plt.figure(figsize=(8, 5))
    for name in names:
        stacked = np.vstack([result[name] for result in results])
        geomean = np.exp(np.mean(np.log(stacked), axis=0))
        plt.plot(trials, geomean, label=name)
    plt.xlabel("Number of trials")
    plt.ylabel("Best loss")
    plt.yscale("log")
    plt.title(f"Geometric mean over {N_REPS} repetitions")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGE_PATH, dpi=200)
    print(f"saved {IMAGE_PATH}")


if __name__ == "__main__":
    main()
