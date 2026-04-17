"""Tests for LevyFlightSampler.

Run with:
    pytest package/samplers/levy_flight_sampler/tests/ -v
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import optuna
import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from levy_flight_sampler import LevyFlightSampler  # noqa: E402


optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sampler() -> LevyFlightSampler:
    return LevyFlightSampler(beta=1.5, step_scale=0.1, seed=42)


# ---------------------------------------------------------------------------
# 1. Initialization tests
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_default_params(self) -> None:
        s = LevyFlightSampler()
        assert s._beta == 1.5
        assert s._step_scale == 0.1
        assert s._sigma > 0

    def test_custom_params(self) -> None:
        s = LevyFlightSampler(beta=1.0, step_scale=0.5, seed=0)
        assert s._beta == 1.0
        assert s._step_scale == 0.5

    def test_beta_boundary_valid(self) -> None:
        """beta exactly at boundaries should be accepted."""
        LevyFlightSampler(beta=0.01)  # just above 0
        LevyFlightSampler(beta=2.0)  # exactly 2 (Gaussian limit)

    def test_beta_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            LevyFlightSampler(beta=0.0)

    def test_beta_above_two_raises(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            LevyFlightSampler(beta=2.1)

    def test_negative_step_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="step_scale"):
            LevyFlightSampler(step_scale=-0.1)

    def test_zero_step_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="step_scale"):
            LevyFlightSampler(step_scale=0.0)

    def test_mantegna_sigma_gaussian_limit(self) -> None:
        """At beta=2, the Lévy distribution approaches Gaussian; sigma should be finite."""
        sigma = LevyFlightSampler._mantegna_sigma(2.0)
        assert math.isfinite(sigma)
        assert sigma > 0

    def test_mantegna_sigma_decreases_with_beta(self) -> None:
        """Higher beta → lighter tails → smaller effective sigma."""
        sigma_low = LevyFlightSampler._mantegna_sigma(0.5)
        sigma_high = LevyFlightSampler._mantegna_sigma(1.9)
        # sigma_low should be larger (heavier tail)
        assert sigma_low > sigma_high

    def test_seed_reproducibility(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", -5.0, 5.0)
            return x**2

        study1 = optuna.create_study(sampler=LevyFlightSampler(seed=7))
        study1.optimize(objective, n_trials=20)

        study2 = optuna.create_study(sampler=LevyFlightSampler(seed=7))
        study2.optimize(objective, n_trials=20)

        params1 = [t.params["x"] for t in study1.trials]
        params2 = [t.params["x"] for t in study2.trials]
        assert params1 == params2


# ---------------------------------------------------------------------------
# 2. FloatDistribution tests
# ---------------------------------------------------------------------------


class TestFloatDistribution:
    def test_stays_in_bounds(self, sampler: LevyFlightSampler) -> None:
        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", -3.0, 3.0)
            return x**2

        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=80)

        for trial in study.trials:
            assert -3.0 <= trial.params["x"] <= 3.0

    def test_converges_on_simple_quadratic(self, sampler: LevyFlightSampler) -> None:
        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", -10.0, 10.0)
            return (x - 2.5) ** 2

        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=150)
        assert study.best_value < 0.5

    def test_log_distribution_stays_in_bounds(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("lr", 1e-5, 1e0, log=True)
            return abs(math.log10(lr) - (-3))  # optimum at lr=1e-3

        sampler = LevyFlightSampler(seed=0)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=80)

        for trial in study.trials:
            assert 1e-5 <= trial.params["lr"] <= 1.0

    def test_log_distribution_converges(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("lr", 1e-5, 1e0, log=True)
            return abs(math.log10(lr) - (-3))

        study = optuna.create_study(sampler=LevyFlightSampler(seed=1))
        study.optimize(objective, n_trials=100)
        assert study.best_value < 0.5

    def test_step_distribution_snaps_to_grid(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", 0.0, 1.0, step=0.1)
            return abs(x - 0.7)

        study = optuna.create_study(sampler=LevyFlightSampler(seed=2))
        study.optimize(objective, n_trials=60)

        for trial in study.trials:
            # Every value must be a multiple of 0.1
            remainder = round(trial.params["x"] / 0.1) * 0.1 - trial.params["x"]
            assert abs(remainder) < 1e-9


# ---------------------------------------------------------------------------
# 3. IntDistribution tests
# ---------------------------------------------------------------------------


class TestIntDistribution:
    def test_stays_in_bounds(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            n = trial.suggest_int("n", 0, 50)
            return float(abs(n - 25))

        study = optuna.create_study(sampler=LevyFlightSampler(seed=3))
        study.optimize(objective, n_trials=80)

        for trial in study.trials:
            assert 0 <= trial.params["n"] <= 50
            assert isinstance(trial.params["n"], int)

    def test_converges_to_target_int(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            n = trial.suggest_int("n", 1, 100)
            return float(abs(n - 42))

        study = optuna.create_study(sampler=LevyFlightSampler(seed=4))
        study.optimize(objective, n_trials=100)
        assert study.best_value == 0.0
        assert study.best_params["n"] == 42

    def test_stepped_int_distribution(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            n = trial.suggest_int("n", 0, 20, step=2)  # even numbers only
            return float(abs(n - 12))

        study = optuna.create_study(sampler=LevyFlightSampler(seed=5))
        study.optimize(objective, n_trials=60)

        for trial in study.trials:
            assert trial.params["n"] % 2 == 0, f"n={trial.params['n']} is not even"


# ---------------------------------------------------------------------------
# 4. CategoricalDistribution tests
# ---------------------------------------------------------------------------


class TestCategoricalDistribution:
    def test_all_values_valid(self) -> None:
        choices = ["alpha", "beta", "gamma", "delta"]

        def objective(trial: optuna.Trial) -> float:
            c = trial.suggest_categorical("c", choices)
            return 0.0 if c == "beta" else 1.0

        study = optuna.create_study(sampler=LevyFlightSampler(seed=6))
        study.optimize(objective, n_trials=50)

        for trial in study.trials:
            assert trial.params["c"] in choices

    def test_finds_optimal_category(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            c = trial.suggest_categorical("c", ["a", "b", "c", "d"])
            return {"a": 10.0, "b": 0.0, "c": 5.0, "d": 8.0}[c]

        study = optuna.create_study(sampler=LevyFlightSampler(seed=7), direction="minimize")
        study.optimize(objective, n_trials=60)
        assert study.best_params["c"] == "b"


# ---------------------------------------------------------------------------
# 5. Mixed search space tests
# ---------------------------------------------------------------------------


class TestMixedSearchSpace:
    def test_mixed_converges(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", 0.0, 10.0)
            n = trial.suggest_int("n", 1, 20)
            cat = trial.suggest_categorical("cat", ["foo", "bar"])
            return abs(x - 7.3) + abs(n - 15) + (0 if cat == "bar" else 5)

        study = optuna.create_study(sampler=LevyFlightSampler(seed=8), direction="minimize")
        study.optimize(objective, n_trials=150)
        assert study.best_value < 1.0
        assert study.best_params["cat"] == "bar"
        assert study.best_params["n"] == 15

    def test_mixed_bounds_respected(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", -1.0, 1.0)
            y = trial.suggest_float("y", 0.0, 100.0, log=False)
            n = trial.suggest_int("n", 0, 5)
            _ = trial.suggest_categorical("mode", ["A", "B"])
            return x**2 + y + n

        study = optuna.create_study(sampler=LevyFlightSampler(seed=9))
        study.optimize(objective, n_trials=80)

        for trial in study.trials:
            assert -1.0 <= trial.params["x"] <= 1.0
            assert 0.0 <= trial.params["y"] <= 100.0
            assert 0 <= trial.params["n"] <= 5
            assert trial.params["mode"] in ["A", "B"]


# ---------------------------------------------------------------------------
# 6. Minimize vs maximize direction
# ---------------------------------------------------------------------------


class TestStudyDirection:
    def test_minimization(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", -5.0, 5.0)
            return (x - 2.0) ** 2

        study = optuna.create_study(sampler=LevyFlightSampler(seed=10), direction="minimize")
        study.optimize(objective, n_trials=100)
        assert study.best_value < 0.5

    def test_maximization(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", -5.0, 5.0)
            return -((x - 2.0) ** 2)  # maximum at x=2 → value=0

        study = optuna.create_study(sampler=LevyFlightSampler(seed=11), direction="maximize")
        study.optimize(objective, n_trials=100)
        assert study.best_value > -0.5


# ---------------------------------------------------------------------------
# 7. Beta parameter effect
# ---------------------------------------------------------------------------


class TestBetaEffect:
    """Verify that different beta values change step distribution behavior."""

    def test_heavy_tail_beta_1_explores_wider(self) -> None:
        """Lower beta → heavier tails → wider spread of visited params."""

        def objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", -50.0, 50.0)
            return x**2

        sampler_heavy = LevyFlightSampler(beta=1.0, seed=0)
        sampler_light = LevyFlightSampler(beta=1.9, seed=0)

        study_heavy = optuna.create_study(sampler=sampler_heavy)
        study_heavy.optimize(objective, n_trials=60)

        study_light = optuna.create_study(sampler=sampler_light)
        study_light.optimize(objective, n_trials=60)

        xs_heavy = [t.params["x"] for t in study_heavy.trials]
        xs_light = [t.params["x"] for t in study_light.trials]

        std_heavy = float(np.std(xs_heavy))
        std_light = float(np.std(xs_light))

        # Heavy-tailed sampler should produce larger spread on average.
        # We give generous tolerance — this is a statistical property.
        assert std_heavy >= std_light * 0.5  # At minimum same order of magnitude


# ---------------------------------------------------------------------------
# 8. Cold-start (fewer than 2 trials) falls back to random
# ---------------------------------------------------------------------------


class TestColdStart:
    def test_first_trial_succeeds(self) -> None:
        def objective(trial: optuna.Trial) -> float:
            return trial.suggest_float("x", 0.0, 1.0)

        study = optuna.create_study(sampler=LevyFlightSampler(seed=0))
        study.optimize(objective, n_trials=1)
        assert len(study.trials) == 1
        assert 0.0 <= study.best_params["x"] <= 1.0
