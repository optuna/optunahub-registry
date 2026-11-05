"""Gibbs sampler for Bayesian linear regression with a horseshoe prior, used by BOCS.

The model follows Baptista & Poloczek (2018), Sect. 3.1, with the half-Cauchy scales
re-parameterized via inverse-gamma auxiliary variables (Makalic & Schmidt, 2016):

    y | alpha, sigma2 ~ N(X @ alpha, sigma2 * I)
    alpha_k | beta2_k, tau2, sigma2 ~ N(0, beta2_k * tau2 * sigma2)
    beta2_k | nu_k ~ IG(1/2, 1/nu_k),  nu_k ~ IG(1/2, 1)
    tau2 | xi ~ IG(1/2, 1/xi),  xi ~ IG(1/2, 1)
    p(sigma2) proportional to 1 / sigma2

The coefficient step draws from

    alpha | ... ~ N(A^{-1} X.T @ y, sigma2 * A^{-1}),
    A = X.T @ X + diag(1 / lambda_diag),

where ``lambda_diag`` contains the diagonal entries of Lambda = tau2 * diag(beta2).

This uses the Bhattacharya et al. (2016) auxiliary-variable algorithm and solves
an N-by-N system, which is advantageous when the number of observations N is
much smaller than the number of coefficients p.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# The fast coefficient sampler is accurate for prior scales in this range (see
# test_fast_alpha_sampler_with_extreme_horseshoe_scales); the scale parameters are
# kept inside it so that 1 / scale never overflows.
_SCALE_MIN = 1e-8
_SCALE_MAX = 1e8


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


@dataclass
class HorseshoeState:
    """Current values of all variables in the Gibbs chain.

    Returned by :func:`sample_horseshoe` so that the next call can continue the
    chain (warm start) instead of burning in again after each new observation.
    """

    alpha: np.ndarray
    sigma2: float
    beta2: np.ndarray
    tau2: float
    nu: np.ndarray
    xi: float

    @classmethod
    def initial(cls, n_coefficients: int) -> HorseshoeState:
        return cls(
            alpha=np.zeros(n_coefficients),
            sigma2=1.0,
            beta2=np.ones(n_coefficients),
            tau2=1.0,
            nu=np.ones(n_coefficients),
            xi=1.0,
        )


def _sample_inverse_gamma(
    shape: float, scale: float | np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    # If G ~ Gamma(shape, 1), then scale / G ~ IG(shape, scale), with density
    # proportional to x^(-shape - 1) * exp(-scale / x).
    scale = np.asarray(scale, dtype=float)
    return scale / rng.gamma(shape, size=scale.shape)


def _prior_scales(state: HorseshoeState) -> np.ndarray:
    return np.clip(state.tau2 * state.beta2, _SCALE_MIN, _SCALE_MAX)


def _update_alpha(
    X: np.ndarray, y: np.ndarray, state: HorseshoeState, rng: np.random.Generator
) -> None:
    n, p = X.shape
    # The fast sampler solves an N-by-N system and the Cholesky sampler a p-by-p one,
    # so use whichever is smaller. N exceeds p late in a run when d is small.
    sample_alpha = sample_alpha_fast if n < p else sample_alpha_cholesky
    state.alpha = sample_alpha(X, y, state.sigma2, _prior_scales(state), rng)


def _update_sigma2(
    X: np.ndarray, y: np.ndarray, state: HorseshoeState, rng: np.random.Generator
) -> None:
    n, p = X.shape
    residual = y - X @ state.alpha
    sigma2_scale = 0.5 * (residual @ residual) + 0.5 * np.sum(
        state.alpha**2 / _prior_scales(state)
    )
    state.sigma2 = float(
        np.clip(_sample_inverse_gamma(0.5 * (n + p), sigma2_scale, rng), _SCALE_MIN, _SCALE_MAX)
    )


def _update_shrinkage(state: HorseshoeState, rng: np.random.Generator) -> None:
    """Update the local and global scales and their auxiliary variables."""
    p = state.alpha.shape[0]
    alpha2 = state.alpha**2

    beta2_scale = 1.0 / state.nu + alpha2 / (2.0 * state.tau2 * state.sigma2)
    state.beta2 = np.clip(_sample_inverse_gamma(1.0, beta2_scale, rng), _SCALE_MIN, _SCALE_MAX)
    state.nu = _sample_inverse_gamma(1.0, 1.0 + 1.0 / state.beta2, rng)

    tau2_scale = 1.0 / state.xi + np.sum(alpha2 / state.beta2) / (2.0 * state.sigma2)
    state.tau2 = float(
        np.clip(_sample_inverse_gamma(0.5 * (p + 1), tau2_scale, rng), _SCALE_MIN, _SCALE_MAX)
    )
    state.xi = float(_sample_inverse_gamma(1.0, 1.0 + 1.0 / state.tau2, rng))


def _gibbs_sweep(
    X: np.ndarray, y: np.ndarray, state: HorseshoeState, rng: np.random.Generator
) -> None:
    """Update every variable once from its full conditional, in place."""
    _update_alpha(X, y, state, rng)
    _update_sigma2(X, y, state, rng)
    _update_shrinkage(state, rng)


def sample_horseshoe(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    *,
    n_burnin: int = 0,
    state: HorseshoeState | None = None,
) -> tuple[np.ndarray, HorseshoeState]:
    """Draw posterior samples of the regression coefficients with Gibbs sampling.

    Parameters
    ----------
    X:
        Design matrix with shape (n_observations, n_coefficients).
    y:
        Response vector with shape (n_observations,).
    n_samples:
        Number of coefficient samples to return.
    rng:
        NumPy random generator, passed in for reproducibility.
    n_burnin:
        Number of sweeps to discard before collecting samples.
    state:
        Chain state to continue from, e.g. the state returned by a previous call
        after one more observation was added. Defaults to a fresh initial state.
        It is not modified; a new state is returned.

    Returns
    -------
    tuple[np.ndarray, HorseshoeState]
        Coefficient samples with shape (n_samples, n_coefficients), and the state
        after the last sweep.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    p = X.shape[1]
    if n_samples < 1 or n_burnin < 0:
        raise ValueError("n_samples must be positive and n_burnin nonnegative.")

    if state is None:
        state = HorseshoeState.initial(p)
    else:
        if state.alpha.shape != (p,):
            raise ValueError(f"state has {state.alpha.shape[0]} coefficients, X has {p}.")
        state = HorseshoeState(
            alpha=state.alpha.copy(),
            sigma2=state.sigma2,
            beta2=state.beta2.copy(),
            tau2=state.tau2,
            nu=state.nu.copy(),
            xi=state.xi,
        )

    for _ in range(n_burnin):
        _gibbs_sweep(X, y, state, rng)

    samples = np.empty((n_samples, p))
    for i in range(n_samples):
        _gibbs_sweep(X, y, state, rng)
        samples[i] = state.alpha
    return samples, state
