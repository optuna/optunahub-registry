"""Tests for the distributionally robust level-set estimation sampler."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import optuna
import optunahub
import pytest


_mod = optunahub.load_local_module(package="samplers/drlse", registry_root="package/")
DRLevelSetSampler = _mod.DRLevelSetSampler
_lse = importlib.import_module(f"{_mod.__name__}._lse")
_drptr = importlib.import_module(f"{_mod.__name__}._drptr")
_acquisition = importlib.import_module(f"{_mod.__name__}._acquisition")

X_POINTS = np.linspace(-2.0, 2.0, 12)
W_POINTS = np.linspace(-1.0, 1.0, 6)


def _objective(trial: optuna.Trial) -> float:
    x = X_POINTS[trial.suggest_int("x_index", 0, len(X_POINTS) - 1)]
    w = W_POINTS[trial.suggest_int("w_index", 0, len(W_POINTS) - 1)]
    return float(np.sin(2.0 * x) + 0.3 * w)


def _make(**kwargs: Any) -> Any:
    params: dict[str, Any] = {"h": 0.0, "alpha": 0.4, "eps": 0.3, "seed": 0}
    params.update(kwargs)
    return DRLevelSetSampler(X_POINTS, W_POINTS, **params)


# --------------------------------------------------------------------------- #
# Sampler behaviour
# --------------------------------------------------------------------------- #


def test_optimization_runs() -> None:
    study = optuna.create_study(sampler=_make())
    study.optimize(_objective, n_trials=20)
    assert len(study.trials) == 20
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)


def test_classification_partitions_the_design_space() -> None:
    sampler = _make()
    study = optuna.create_study(sampler=sampler)
    study.optimize(_objective, n_trials=25)
    reliable, unreliable, unclassified = sampler.classify(study)
    assert reliable.shape == unreliable.shape == unclassified.shape == X_POINTS.shape
    assert not (reliable & unreliable).any()
    assert (reliable | unreliable | unclassified).all()


def test_reliable_set_is_a_subset_of_the_design_points() -> None:
    sampler = _make()
    study = optuna.create_study(sampler=sampler)
    study.optimize(_objective, n_trials=20)
    assert np.isin(sampler.reliable_set(study), X_POINTS).all()


def test_reproducibility() -> None:
    runs = []
    for _ in range(2):
        study = optuna.create_study(sampler=_make())
        study.optimize(_objective, n_trials=15)
        runs.append([(t.params["x_index"], t.params["w_index"]) for t in study.trials])
    assert runs[0] == runs[1]


def test_no_point_is_evaluated_twice() -> None:
    sampler = _make(n_startup_trials=3)
    study = optuna.create_study(sampler=sampler)
    study.optimize(_objective, n_trials=30)
    seen = [(t.params["x_index"], t.params["w_index"]) for t in study.trials]
    assert len(set(seen)) == len(seen)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": 1.5},
        {"alpha": 0.0},
        {"eps": -0.1},
        {"eta": -1.0},
        {"noise": 0.0},
        {"n_startup_trials": 0},
        {"p_ref": np.full(len(W_POINTS), 0.5)},
    ],
)
def test_invalid_arguments(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _make(**kwargs)


# --------------------------------------------------------------------------- #
# The DRPTR transport problem
# --------------------------------------------------------------------------- #


def test_drptr_matches_the_linear_program() -> None:
    """The closed form must agree with the LP formulation of eq. (3.1)."""
    rng = np.random.default_rng(0)
    for _ in range(120):
        n = int(rng.integers(2, 12))
        p = rng.random(n)
        p /= p.sum()
        c = rng.random(n)
        eps = float(rng.uniform(0.0, 1.5))
        assert _drptr.drptr(c, p, eps) == pytest.approx(_drptr.drptr_linprog(c, p, eps), abs=1e-12)


def test_drptr_binary_matches_the_general_form() -> None:
    rng = np.random.default_rng(1)
    for _ in range(120):
        n = int(rng.integers(2, 12))
        p = rng.random(n)
        p /= p.sum()
        ind = rng.integers(0, 2, size=n).astype(float)
        eps = float(rng.uniform(0.0, 1.5))
        assert _drptr.drptr_binary(ind, p, eps, float(ind @ p)) == pytest.approx(
            _drptr.drptr(ind, p, eps), abs=1e-12
        )


def test_drptr_edge_cases() -> None:
    p = np.array([0.2, 0.5, 0.3])
    c = np.array([1.0, 0.0, 1.0])
    assert _drptr.drptr(c, p, 0.0) == pytest.approx(0.5)  # no budget: reference expectation
    assert _drptr.drptr(c, p, 2.0) == pytest.approx(0.0)  # full budget: minimum cost
    assert _drptr.drptr(np.ones(3), p, 1.0) == pytest.approx(1.0)  # nowhere to move mass
    with pytest.raises(ValueError):
        _drptr.drptr(np.array([]), np.array([]), 0.5)


# --------------------------------------------------------------------------- #
# Estimator and acquisition
# --------------------------------------------------------------------------- #


def _demo_estimator(eta: float = 0.0) -> Any:
    kernel, _ = _lse.gaussian_kernel(X_POINTS, W_POINTS, 1.0, 1.0)
    return _lse.DRLevelSetEstimator(
        kernel,
        len(X_POINTS),
        len(W_POINTS),
        np.full(len(W_POINTS), 1 / len(W_POINTS)),
        0.3,
        0.0,
        0.4,
        eta=eta,
    )


def test_eta_zero_never_misclassifies() -> None:
    """With eta = 0 the credible bounds are sound, so H and L cannot contradict the truth."""
    rng = np.random.default_rng(3)
    kernel, _ = _lse.gaussian_kernel(X_POINTS, W_POINTS, 1.0, 1.0)
    p_ref = np.full(len(W_POINTS), 1 / len(W_POINTS))
    chol = np.linalg.cholesky(kernel + 1e-8 * np.eye(len(kernel)))
    f = (chol @ rng.standard_normal(len(kernel))).reshape(len(X_POINTS), len(W_POINTS))
    truth = _lse.true_drptr(f, p_ref, 0.3, 0.0) > 0.4

    est = _demo_estimator()
    for step, flat in enumerate(rng.permutation(f.size)):
        est.observe(int(flat) // len(W_POINTS), int(flat) % len(W_POINTS), float(f.ravel()[flat]))
        if step % 12 == 0:
            reliable, unreliable, _ = est.classify(4.0)
            assert not (reliable & ~truth).any()
            assert not (unreliable & truth).any()


def test_grid_indices_are_bounds_checked() -> None:
    est = _demo_estimator()
    with pytest.raises(IndexError):
        est.observe(0, len(W_POINTS), 1.0)
    with pytest.raises(IndexError):
        est.observe(0, -1, 1.0)
    with pytest.raises(IndexError):
        est.observe(len(X_POINTS), 0, 1.0)


def test_negative_beta_is_rejected() -> None:
    with pytest.raises(ValueError):
        _demo_estimator().classify(-1.0)


def test_zeta_bounds_the_acquisition_error() -> None:
    """Lemma 3.3: dropping low-probability regions costs at most zeta per design point."""
    rng = np.random.default_rng(7)
    kernel, _ = _lse.gaussian_kernel(X_POINTS, W_POINTS, 1.0, 1.0)
    p_ref = np.full(len(W_POINTS), 1 / len(W_POINTS))
    est = _lse.DRLevelSetEstimator(
        kernel, len(X_POINTS), len(W_POINTS), p_ref, 0.3, 0.0, 0.4, eta=0.0
    )
    chol = np.linalg.cholesky(kernel + 1e-8 * np.eye(len(kernel)))
    f = (chol @ rng.standard_normal(len(kernel))).ravel()
    for flat in rng.permutation(f.size)[:20]:
        est.observe(int(flat) // len(W_POINTS), int(flat) % len(W_POINTS), float(f[flat]))

    exact, unclassified = _acquisition.acquisition_values(est, 4.0, gamma=0.0, zeta=0.0)
    for zeta in (1e-6, 1e-3, 1e-2):
        approx, _ = _acquisition.acquisition_values(est, 4.0, gamma=0.0, zeta=zeta)
        assert np.abs(approx - exact).max() <= unclassified.size * zeta + 1e-12
        assert (approx <= exact + 1e-12).all()  # dropped mass is never credited

    with pytest.raises(ValueError):
        _acquisition.acquisition_values(est, 4.0, zeta=-1.0)


def test_acquisition_is_finite_on_a_widely_separated_grid() -> None:
    """Regression: a design row with no live breakpoints once scored a spurious +1.

    A representative point taken by offsetting from an infinite region edge evaluated to
    infinity, which drove the indicator to all-ones and credited a full classification to
    the candidate furthest from every observation.
    """
    x_points = np.array([0.0, 0.3, 0.6, 8.0])
    w_points = np.array([-0.3, 0.0, 0.3])
    kernel, _ = _lse.gaussian_kernel(x_points, w_points, 1.0, 1.0)
    est = _lse.DRLevelSetEstimator(
        kernel, 4, 3, np.full(3, 1 / 3), 0.3, 0.0, 0.4, eta=0.0, noise=1e-6
    )
    for j in range(3):
        est.observe(0, j, 0.5)
    values, unclassified = _acquisition.acquisition_values(
        est, 4.0, gamma=0.0, use_variance_term=False
    )
    assert np.isfinite(values).all()
    assert (values <= unclassified.size + 1e-9).all()
    far = values[9:12]
    near = values[3:9]
    assert far.max() < near.max(), "the most distant candidate must not win"
