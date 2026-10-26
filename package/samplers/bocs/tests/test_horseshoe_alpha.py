from __future__ import annotations

import numpy as np
import pytest

from .._horseshoe import sample_alpha_cholesky
from .._horseshoe import sample_alpha_fast


def test_fast_alpha_sampler_matches_cholesky_mean_and_covariance() -> None:
    """Empirical moments from both samplers should match the same target."""
    rng_data = np.random.default_rng(20261025)
    n, p = 5, 9  # Deliberately use n < p to exercise the fast formulation.
    X = rng_data.normal(size=(n, p))
    y = rng_data.normal(size=n)
    sigma2 = 1.7
    lambda_diag = np.exp(rng_data.normal(size=p))

    A = X.T @ X + np.diag(1.0 / lambda_diag)
    expected_mean = np.linalg.solve(A, X.T @ y)
    expected_cov = sigma2 * np.linalg.solve(A, np.eye(p))

    n_draws = 50_000
    rng_fast = np.random.default_rng(12345)
    rng_chol = np.random.default_rng(67890)

    fast_draws = np.empty((n_draws, p))
    chol_draws = np.empty((n_draws, p))
    for i in range(n_draws):
        fast_draws[i] = sample_alpha_fast(X, y, sigma2, lambda_diag, rng_fast)
        chol_draws[i] = sample_alpha_cholesky(X, y, sigma2, lambda_diag, rng_chol)

    fast_mean = fast_draws.mean(axis=0)
    chol_mean = chol_draws.mean(axis=0)
    fast_cov = np.cov(fast_draws, rowvar=False)
    chol_cov = np.cov(chol_draws, rowvar=False)

    # Monte Carlo tolerances: compare both empirical moments to the exact
    # target, rather than requiring two random estimates to be identical.
    mean_se = np.sqrt(np.diag(expected_cov) / n_draws)
    assert np.all(np.abs(fast_mean - expected_mean) < 5.0 * mean_se)
    assert np.all(np.abs(chol_mean - expected_mean) < 5.0 * mean_se)

    # Monte Carlo standard error of each sample-covariance entry for Gaussian
    # draws: Var(C_hat_ij) ~= (C_ii * C_jj + C_ij**2) / n_draws.
    diag = np.diag(expected_cov)
    cov_se = np.sqrt((np.outer(diag, diag) + expected_cov**2) / n_draws)
    assert np.all(np.abs(fast_cov - expected_cov) < 5.0 * cov_se)
    assert np.all(np.abs(chol_cov - expected_cov) < 5.0 * cov_se)


def test_fast_alpha_sampler_is_reproducible() -> None:
    rng_data = np.random.default_rng(1)
    X = rng_data.normal(size=(4, 7))
    y = rng_data.normal(size=4)
    lambda_diag = np.ones(7)

    first = sample_alpha_fast(X, y, 1.0, lambda_diag, np.random.default_rng(42))
    second = sample_alpha_fast(X, y, 1.0, lambda_diag, np.random.default_rng(42))
    np.testing.assert_array_equal(first, second)


def _assert_fast_mean_matches_target(
    X: np.ndarray, y: np.ndarray, sigma2: float, lambda_diag: np.ndarray, n_draws: int
) -> None:
    A = X.T @ X + np.diag(1.0 / lambda_diag)
    expected_mean = np.linalg.solve(A, X.T @ y)
    expected_var = sigma2 * np.diag(np.linalg.inv(A))

    rng = np.random.default_rng(0)
    draws = np.array([sample_alpha_fast(X, y, sigma2, lambda_diag, rng) for _ in range(n_draws)])
    assert np.all(np.isfinite(draws))
    mean_se = np.sqrt(expected_var / n_draws)
    assert np.all(np.abs(draws.mean(axis=0) - expected_mean) < 5.0 * mean_se)


def test_fast_alpha_sampler_with_more_observations_than_coefficients() -> None:
    # Late in an optimization run with small d, N can exceed p.
    rng_data = np.random.default_rng(7)
    n, p = 30, 6
    X = rng_data.normal(size=(n, p))
    y = rng_data.normal(size=n)
    lambda_diag = np.exp(rng_data.normal(size=p))
    _assert_fast_mean_matches_target(X, y, 0.8, lambda_diag, n_draws=20_000)


def test_fast_alpha_sampler_with_extreme_horseshoe_scales() -> None:
    # BOCS-sized problem (d = 20 gives p = 211) with binary features and prior
    # scales spanning 16 orders of magnitude, as the horseshoe can produce.
    rng_data = np.random.default_rng(3)
    n, p = 15, 211
    X = rng_data.integers(0, 2, size=(n, p)).astype(float)
    X[:, 0] = 1.0
    y = rng_data.normal(size=n)
    lambda_diag = 10.0 ** rng_data.uniform(-8.0, 8.0, size=p)
    _assert_fast_mean_matches_target(X, y, 1.0, lambda_diag, n_draws=5_000)


@pytest.mark.parametrize(
    "X, y, sigma2, lambda_diag",
    [
        (np.ones(3), np.ones(3), 1.0, np.ones(3)),  # X not 2-D
        (np.ones((3, 2)), np.ones(4), 1.0, np.ones(2)),  # y shape mismatch
        (np.ones((3, 2)), np.ones(3), 1.0, np.ones(3)),  # lambda shape mismatch
        (np.ones((3, 2)), np.ones(3), 0.0, np.ones(2)),  # sigma2 not positive
        (np.ones((3, 2)), np.ones(3), 1.0, np.array([1.0, 0.0])),  # lambda not positive
        (np.full((3, 2), np.nan), np.ones(3), 1.0, np.ones(2)),  # non-finite X
    ],
)
def test_fast_alpha_sampler_rejects_invalid_inputs(
    X: np.ndarray, y: np.ndarray, sigma2: float, lambda_diag: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        sample_alpha_fast(X, y, sigma2, lambda_diag, np.random.default_rng(0))
