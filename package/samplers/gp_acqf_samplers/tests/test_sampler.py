"""Tests for GP-based samplers with alternative acquisition functions.

Uses ``optuna.testing.pytest_samplers`` (Optuna 4.8+) to reuse the built-in
test cases for Optuna samplers. Custom tests below cover sampler-specific
behaviour (UCB beta effect, MO fallback, constraint fallback, reproducibility).

GPEISampler is an alias of ``optuna.samplers.GPSampler`` so its behaviour is
already covered by Optuna's own test suite; we only assert the alias.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.testing.pytest_samplers import BasicSamplerTestCase
from optuna.testing.pytest_samplers import MultiObjectiveSamplerTestCase
from optuna.testing.pytest_samplers import RelativeSamplerTestCase
import optunahub
import pytest


_mod = optunahub.load_local_module(package="samplers/gp_acqf_samplers", registry_root="package/")
GPEISampler = _mod.GPEISampler
GPPISampler = _mod.GPPISampler
GPUCBSampler = _mod.GPUCBSampler
GPTSSampler = _mod.GPTSSampler


# The generic test cases run the samplers with very few completed trials, so
# we set ``n_startup_trials=1`` in the fixtures to exercise the GP code path
# rather than falling back to random startup sampling.


class TestGPPISampler(
    BasicSamplerTestCase, RelativeSamplerTestCase, MultiObjectiveSamplerTestCase
):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return lambda: GPPISampler(n_startup_trials=1, seed=42)


class TestGPUCBSampler(
    BasicSamplerTestCase, RelativeSamplerTestCase, MultiObjectiveSamplerTestCase
):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return lambda: GPUCBSampler(n_startup_trials=1, seed=42)


class TestGPTSSampler(
    BasicSamplerTestCase, RelativeSamplerTestCase, MultiObjectiveSamplerTestCase
):
    @pytest.fixture
    def sampler(self) -> Callable[[], BaseSampler]:
        return lambda: GPTSSampler(n_startup_trials=1, seed=42)


# --------------------------------------------------------------------------- #
# Sampler-specific custom tests (not covered by the generic test cases)
# --------------------------------------------------------------------------- #


def _sphere(trial: optuna.Trial) -> float:
    return sum(trial.suggest_float(f"x{i}", -5, 5) ** 2 for i in range(3))


def _bi_objective(trial: optuna.Trial) -> tuple[float, float]:
    x = [trial.suggest_float(f"x{i}", -5, 5) for i in range(3)]
    return sum(v**2 for v in x), -sum(v**2 for v in x)


def test_gp_ei_sampler_is_gp_sampler() -> None:
    assert GPEISampler is optuna.samplers.GPSampler


def test_reproducibility() -> None:
    results: list[list[float]] = []
    for _ in range(2):
        sampler = GPPISampler(seed=42)
        study = optuna.create_study(sampler=sampler)
        study.optimize(_sphere, n_trials=15)
        results.append([t.value for t in study.trials])

    np.testing.assert_array_equal(results[0], results[1])


def test_ucb_beta_effect() -> None:
    """Smoke test: UCB runs end-to-end for both low and high beta values."""
    study_low = optuna.create_study(sampler=GPUCBSampler(beta=0.01, seed=42))
    study_low.optimize(_sphere, n_trials=30)

    study_high = optuna.create_study(sampler=GPUCBSampler(beta=10.0, seed=42))
    study_high.optimize(_sphere, n_trials=30)

    assert len(study_low.trials) == 30
    assert len(study_high.trials) == 30


@pytest.mark.parametrize("cls", [GPPISampler, GPUCBSampler, GPTSSampler])
def test_multi_objective_fallback(cls: type[BaseSampler]) -> None:
    """Multi-objective studies must fall back to GPSampler behaviour."""
    sampler = cls(n_startup_trials=1, seed=42)
    study = optuna.create_study(directions=["minimize", "maximize"], sampler=sampler)
    study.optimize(_bi_objective, n_trials=8)

    assert len(study.trials) == 8
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_constraints_func_forwarding() -> None:
    """Verify constraints_func is accepted and triggers fallback to GPSampler."""
    sampler = GPUCBSampler(beta=2.0, seed=42, constraints_func=lambda t: [0.0])
    study = optuna.create_study(sampler=sampler)
    study.optimize(_sphere, n_trials=15)

    assert len(study.trials) == 15
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_ts_constraints_func_forwarding() -> None:
    sampler = GPTSSampler(seed=42, constraints_func=lambda t: [0.0])
    study = optuna.create_study(sampler=sampler)
    study.optimize(_sphere, n_trials=15)

    assert len(study.trials) == 15
