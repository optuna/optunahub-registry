"""Fast Gaussian coefficient sampler used by the BOCS horseshoe sampler.

The sampler draws from

    alpha | ... ~ N(A^{-1} X.T @ y, sigma2 * A^{-1}),
    A = X.T @ X + diag(1 / lambda_diag),

where ``lambda_diag`` contains the diagonal entries of Lambda = tau2 * diag(beta2).

This uses the Bhattacharya et al. (2016) auxiliary-variable algorithm and solves
an N-by-N system, which is advantageous when the number of observations N is
much smaller than the number of coefficients p.
"""

from __future__ import annotations

import numpy as np


def sample_alpha_fast(
    X: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    lambda_diag: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one exact conditional sample of alpha using an N-by-N solve.

    Parameters
    ----------
    X:
        Design matrix with shape (n_observations, n_coefficients).
    y:
        Response vector with shape (n_observations,).
    sigma2:
        Positive observation-noise variance.
    lambda_diag:
        Positive diagonal of Lambda = tau2 * diag(beta2), shape (p,).
    rng:
        NumPy random generator, passed in for reproducibility.

    Returns
    -------
    np.ndarray
        A draw with conditional mean A^{-1} X.T @ y and covariance
        sigma2 * A^{-1}, where A = X.T @ X + diag(1 / lambda_diag).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    lambda_diag = np.asarray(lambda_diag, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    n, p = X.shape
    if y.shape != (n,):
        raise ValueError(f"y must have shape ({n},).")
    if lambda_diag.shape != (p,):
        raise ValueError(f"lambda_diag must have shape ({p},).")
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("sigma2 must be finite and positive.")
    if np.any(~np.isfinite(lambda_diag)) or np.any(lambda_diag <= 0):
        raise ValueError("lambda_diag entries must be finite and positive.")
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(y)):
        raise ValueError("X and y must contain only finite values.")

    # Rescale to the unit-noise model:
    # Phi = X / sqrt(sigma2), a = y / sqrt(sigma2), D = sigma2 * Lambda.
    sqrt_sigma2 = np.sqrt(sigma2)
    Phi = X / sqrt_sigma2
    a = y / sqrt_sigma2
    D_diag = sigma2 * lambda_diag

    # Draw u ~ N(0, D) and delta ~ N(0, I_n).
    u = np.sqrt(D_diag) * rng.standard_normal(p)
    delta = rng.standard_normal(n)
    v = Phi @ u + delta

    # Solve the smaller n-by-n system.
    system = (Phi * D_diag) @ Phi.T + np.eye(n)
    rhs = a - v
    w = np.linalg.solve(system, rhs)

    # D is diagonal, so multiplication by D is elementwise.
    return u + D_diag * (Phi.T @ w)


def sample_alpha_cholesky(
    X: np.ndarray,
    y: np.ndarray,
    sigma2: float,
    lambda_diag: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Reference draw using a p-by-p Cholesky factorization.

    Intended primarily for testing the fast sampler.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    lambda_diag = np.asarray(lambda_diag, dtype=float).reshape(-1)

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    n, p = X.shape
    if y.shape != (n,):
        raise ValueError(f"y must have shape ({n},).")
    if lambda_diag.shape != (p,):
        raise ValueError(f"lambda_diag must have shape ({p},).")
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("sigma2 must be finite and positive.")
    if np.any(~np.isfinite(lambda_diag)) or np.any(lambda_diag <= 0):
        raise ValueError("lambda_diag entries must be finite and positive.")

    A = X.T @ X + np.diag(1.0 / lambda_diag)
    mean = np.linalg.solve(A, X.T @ y)
    covariance = sigma2 * np.linalg.solve(A, np.eye(p))

    # Symmetrize away tiny floating-point asymmetry before Cholesky.
    covariance = 0.5 * (covariance + covariance.T)
    L = np.linalg.cholesky(covariance)
    return mean + L @ rng.standard_normal(p)
