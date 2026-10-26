from __future__ import annotations

import numpy as np

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

    # Elementwise relative errors are unstable for off-diagonal entries whose
    # true covariance is close to zero. Use a global absolute Monte Carlo bound.
    assert np.max(np.abs(fast_cov - expected_cov)) < 0.06
    assert np.max(np.abs(chol_cov - expected_cov)) < 0.06


def test_fast_alpha_sampler_is_reproducible() -> None:
    rng_data = np.random.default_rng(1)
    X = rng_data.normal(size=(4, 7))
    y = rng_data.normal(size=4)
    lambda_diag = np.ones(7)

    first = sample_alpha_fast(X, y, 1.0, lambda_diag, np.random.default_rng(42))
    second = sample_alpha_fast(X, y, 1.0, lambda_diag, np.random.default_rng(42))
    np.testing.assert_array_equal(first, second)
