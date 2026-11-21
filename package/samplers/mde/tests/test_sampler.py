"""Tests for ``MDESampler``. Run from the repository root: ``pytest package/samplers/mde``."""

from __future__ import annotations

import math
import os

import numpy as np
import optuna
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState
import optunahub
import pytest


MDESampler = optunahub.load_local_module(
    package="samplers/mde",
    registry_root=os.path.join(os.path.dirname(__file__), "..", "..", ".."),
).MDESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _sphere(trial: optuna.Trial) -> float:
    x = np.array([trial.suggest_float(f"x{i}", -5.0, 5.0) for i in range(5)])
    return float(np.sum(x**2))


def test_population_size_validation() -> None:
    with pytest.raises(ValueError):
        MDESampler(population_size=2)


def test_generation_structure() -> None:
    n = 5
    sampler = MDESampler(
        search_space={f"x{i}": FloatDistribution(-5.0, 5.0) for i in range(5)},
        population_size=n,
        seed=0,
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(_sphere, n_trials=3 * n)
    assert sampler._generation == 1  # The initial population (3N trials) is finished.
    study.optimize(_sphere, n_trials=n)
    assert sampler._generation == 2


def test_beats_random_search() -> None:
    mde = optuna.create_study(sampler=MDESampler(population_size=10, seed=0))
    mde.optimize(_sphere, n_trials=600)
    rnd = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    rnd.optimize(_sphere, n_trials=600)
    assert mde.best_value < rnd.best_value


def test_reproducible_with_seed() -> None:
    values = []
    for _ in range(2):
        study = optuna.create_study(sampler=MDESampler(population_size=10, seed=123))
        study.optimize(_sphere, n_trials=300)
        values.append([t.value for t in study.trials])
    assert values[0] == values[1]


def test_maximize() -> None:
    def objective(trial: optuna.Trial) -> float:
        return -_sphere(trial)

    study = optuna.create_study(direction="maximize", sampler=MDESampler(seed=0))
    study.optimize(objective, n_trials=600)
    first_generation_best = max(t.value for t in study.trials[:90] if t.value is not None)
    assert study.best_value > first_generation_best


def test_mixed_search_space() -> None:
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -5.0, 5.0)
        n = trial.suggest_int("n", -10, 10)
        k = trial.suggest_int("k", 2, 64, step=2)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        m = trial.suggest_int("m", 1, 1000, log=True)
        s = trial.suggest_float("s", 0.0, 1.0, step=0.1)
        c = trial.suggest_categorical("c", ["a", "b", "c"])
        assert isinstance(n, int) and isinstance(k, int) and k % 2 == 0
        assert isinstance(m, int) and 1 <= m <= 1000
        assert 1e-5 <= lr <= 1e-1
        assert abs(round(s * 10) - s * 10) < 1e-8
        return x**2 + n**2 + (k - 32) ** 2 + math.log10(lr) ** 2 + s + {"a": 0, "b": 1, "c": 2}[c]

    study = optuna.create_study(sampler=MDESampler(population_size=5, seed=0))
    study.optimize(objective, n_trials=200)
    assert len(study.trials) == 200


def test_pruned_and_failed_trials() -> None:
    def objective(trial: optuna.Trial) -> float:
        value = _sphere(trial)
        if trial.number % 7 == 3:
            raise optuna.TrialPruned()
        if trial.number % 11 == 5:
            raise RuntimeError("failure")
        return value

    sampler = MDESampler(population_size=5, seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=200, catch=(RuntimeError,))
    states = {t.state for t in study.trials}
    assert TrialState.PRUNED in states and TrialState.FAIL in states
    assert sampler._generation > 10
    assert np.isfinite(sampler._fitA).any() and np.isfinite(sampler._best).all()


def test_parallel() -> None:
    sampler = MDESampler(population_size=10, seed=0)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_sphere, n_trials=400, n_jobs=4)
    assert len(study.trials) == 400
    assert all(t.state == TrialState.COMPLETE for t in study.trials)
    assert sampler._generation > 10


def test_ask_tell() -> None:
    study = optuna.create_study(sampler=MDESampler(population_size=5, seed=0))
    for _ in range(100):
        trial = study.ask()
        study.tell(trial, _sphere(trial))
    assert len(study.trials) == 100


def test_multi_objective_not_supported() -> None:
    def objective(trial: optuna.Trial) -> tuple[float, float]:
        x = trial.suggest_float("x", 0, 1)
        return x, 1 - x

    study = optuna.create_study(directions=["minimize", "minimize"], sampler=MDESampler())
    with pytest.raises(ValueError):
        study.optimize(objective, n_trials=3)
