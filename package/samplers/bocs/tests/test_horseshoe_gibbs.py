from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from .._horseshoe import _sample_inverse_gamma
from .._horseshoe import _update_alpha
from .._horseshoe import _update_shrinkage
from .._horseshoe import HorseshoeState
from .._horseshoe import sample_horseshoe


def _second_order_features(x: np.ndarray) -> np.ndarray:
    # [1, x_1, ..., x_d, x_i * x_j for i < j], the BOCS design matrix.
    pairs = [x[:, i] * x[:, j] for i, j in combinations(range(x.shape[1]), 2)]
    return np.column_stack([np.ones(len(x)), x, *pairs])


def _sparse_problem(
    n: int, noise_std: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = _second_order_features(rng.integers(0, 2, size=(n, 8)).astype(float))
    alpha_true = np.zeros(X.shape[1])
    alpha_true[[0, 3, 15]] = [1.0, 3.0, -2.0]  # intercept, x_3, one interaction
    y = X @ alpha_true + noise_std * rng.standard_normal(n)
    return X, y, alpha_true


def test_inverse_gamma_matches_analytic_moments() -> None:
    # IG(a, b) has mean b / (a - 1) and variance b^2 / ((a - 1)^2 (a - 2)).
    a, b, n = 4.0, 3.0, 200_000
    draws = _sample_inverse_gamma(a, np.full(n, b), np.random.default_rng(0))
    mean = b / (a - 1.0)
    std = b / ((a - 1.0) * np.sqrt(a - 2.0))
    assert abs(draws.mean() - mean) < 5.0 * std / np.sqrt(n)
    assert np.all(draws > 0)


def test_posterior_recovers_sparse_coefficients() -> None:
    X, y, alpha_true = _sparse_problem(n=60, noise_std=0.1, seed=0)
    samples, _ = sample_horseshoe(X, y, 2_000, np.random.default_rng(1), n_burnin=500)
    posterior_mean = samples.mean(axis=0)

    nonzero = alpha_true != 0
    np.testing.assert_allclose(posterior_mean[nonzero], alpha_true[nonzero], atol=0.15)
    # The horseshoe should shrink the 34 irrelevant coefficients toward zero.
    assert np.max(np.abs(posterior_mean[~nonzero])) < 0.15


def test_posterior_noise_variance_is_close_to_truth() -> None:
    X, y, _ = _sparse_problem(n=300, noise_std=0.5, seed=2)
    rng = np.random.default_rng(3)
    state = HorseshoeState.initial(X.shape[1])
    sigma2_draws = []
    for _ in range(1_000):
        _, state = sample_horseshoe(X, y, 1, rng, state=state)
        sigma2_draws.append(state.sigma2)
    assert np.mean(sigma2_draws[200:]) == pytest.approx(0.25, rel=0.2)


def test_shrinkage_conditionals_get_it_right() -> None:
    """Geweke (2004) "getting it right" test for the shrinkage updates.

    Alternating the parameter updates with fresh draws of y from the likelihood
    leaves the joint prior p(theta) p(y | theta) invariant, so the scales must keep
    their half-Cauchy prior, whose median is 1. A wrong full conditional shifts these
    distributions. sigma2 is held fixed because its prior is improper.
    """
    rng = np.random.default_rng(1)
    n, p, n_iter = 3, 4, 50_000
    X = rng.normal(size=(n, p))
    state = HorseshoeState.initial(p)
    y = rng.standard_normal(n)

    tau = np.empty(n_iter)
    beta = np.empty((n_iter, p))
    for t in range(n_iter):
        _update_alpha(X, y, state, rng)
        _update_shrinkage(state, rng)
        y = X @ state.alpha + np.sqrt(state.sigma2) * rng.standard_normal(n)
        tau[t] = np.sqrt(state.tau2)
        beta[t] = np.sqrt(state.beta2)

    burnin = n_iter // 10
    # Tolerances allow for autocorrelation; the global scale tau mixes more slowly.
    assert abs(np.mean(beta[burnin:] < 1.0) - 0.5) < 0.05
    assert abs(np.mean(tau[burnin:] < 1.0) - 0.5) < 0.12


def test_sampler_is_reproducible() -> None:
    X, y, _ = _sparse_problem(n=20, noise_std=0.1, seed=4)
    first, _ = sample_horseshoe(X, y, 50, np.random.default_rng(5), n_burnin=10)
    second, _ = sample_horseshoe(X, y, 50, np.random.default_rng(5), n_burnin=10)
    np.testing.assert_array_equal(first, second)


def test_warm_start_continues_chain_without_mutating_input_state() -> None:
    X, y, _ = _sparse_problem(n=20, noise_std=0.1, seed=6)
    rng = np.random.default_rng(7)
    _, state = sample_horseshoe(X, y, 10, rng, n_burnin=10)
    alpha_before = state.alpha.copy()

    samples, new_state = sample_horseshoe(X, y, 5, rng, state=state)

    np.testing.assert_array_equal(state.alpha, alpha_before)
    assert new_state is not state
    assert samples.shape == (5, X.shape[1])
    assert np.all(np.isfinite(samples))
    assert new_state.sigma2 > 0 and new_state.tau2 > 0 and np.all(new_state.beta2 > 0)


def test_rejects_state_with_wrong_number_of_coefficients() -> None:
    X, y, _ = _sparse_problem(n=20, noise_std=0.1, seed=8)
    with pytest.raises(ValueError):
        sample_horseshoe(X, y, 1, np.random.default_rng(0), state=HorseshoeState.initial(3))
