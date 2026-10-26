"""Second-order model over binary inputs used by BOCS.

For x in {0, 1}^d the model is (Baptista & Poloczek, 2018, Eq. 1)

    f(x) = alpha_0 + sum_j alpha_j x_j + sum_{i<j} alpha_ij x_i x_j,

which is linear in the coefficients alpha, so it is fit as a linear regression on
the features [1, x_1, ..., x_d, x_1 x_2, x_1 x_3, ..., x_{d-1} x_d].
"""

from __future__ import annotations

import numpy as np


def n_coefficients(n_vars: int) -> int:
    """Number of coefficients p = 1 + d + d (d - 1) / 2 of the second-order model."""
    return 1 + n_vars + n_vars * (n_vars - 1) // 2


def second_order_features(x: np.ndarray) -> np.ndarray:
    """Map binary inputs of shape (n, d) to model features of shape (n, p)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    rows, cols = np.triu_indices(x.shape[1], k=1)
    return np.hstack([np.ones((x.shape[0], 1)), x, x[:, rows] * x[:, cols]])


def quadratic_form(alpha: np.ndarray, n_vars: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Rewrite the model as f(x) = c + b @ x + x @ A @ x with symmetric A.

    A has a zero diagonal, because x_i^2 = x_i for binary x and the model has no
    separate squared terms.
    """
    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (n_coefficients(n_vars),):
        raise ValueError(f"alpha must have shape ({n_coefficients(n_vars)},).")
    constant = float(alpha[0])
    b = alpha[1 : n_vars + 1].copy()
    A = np.zeros((n_vars, n_vars))
    rows, cols = np.triu_indices(n_vars, k=1)
    # Each pair appears twice in x @ A @ x, so split its coefficient in half.
    A[rows, cols] = 0.5 * alpha[n_vars + 1 :]
    A[cols, rows] = A[rows, cols]
    return constant, b, A
