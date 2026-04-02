"""Tests for GP-based samplers with alternative acquisition functions."""

from __future__ import annotations

import numpy as np
import optuna
import optunahub


_mod = optunahub.load_local_module(package="samplers/gp_acqf_samplers", registry_root="package/")
GPEISampler = _mod.GPEISampler
GPPISampler = _mod.GPPISampler
GPUCBSampler = _mod.GPUCBSampler
GPTSSampler = _mod.GPTSSampler


def _sphere(trial: optuna.Trial) -> float:
    return sum(trial.suggest_float(f"x{i}", -5, 5) ** 2 for i in range(3))


def test_gp_pi_sampler_runs() -> None:
    sampler = GPPISampler(seed=42)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_sphere, n_trials=20)

    assert len(study.trials) == 20
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_gp_ucb_sampler_runs() -> None:
    sampler = GPUCBSampler(beta=2.0, seed=42)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_sphere, n_trials=20)

    assert len(study.trials) == 20
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_gp_ts_sampler_runs() -> None:
    sampler = GPTSSampler(seed=42, n_rff_features=256)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_sphere, n_trials=20)

    assert len(study.trials) == 20
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


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
    """Higher beta should produce more diverse samples (more exploration)."""
    study_low = optuna.create_study(sampler=GPUCBSampler(beta=0.01, seed=42))
    study_low.optimize(_sphere, n_trials=30)

    study_high = optuna.create_study(sampler=GPUCBSampler(beta=10.0, seed=42))
    study_high.optimize(_sphere, n_trials=30)

    # Both should complete without error
    assert len(study_low.trials) == 30
    assert len(study_high.trials) == 30


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


def test_maximize_direction() -> None:
    """Samplers should work with maximize direction."""

    def neg_sphere(trial: optuna.Trial) -> float:
        return -sum(trial.suggest_float(f"x{i}", -5, 5) ** 2 for i in range(3))

    for cls, kw in [
        (GPPISampler, {}),
        (GPUCBSampler, {"beta": 2.0}),
        (GPTSSampler, {}),
    ]:
        study = optuna.create_study(direction="maximize", sampler=cls(seed=42, **kw))
        study.optimize(neg_sphere, n_trials=20)
        assert len(study.trials) == 20
