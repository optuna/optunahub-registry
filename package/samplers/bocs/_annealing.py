"""Simulated-annealing acquisition optimizer of BOCS-SA (Baptista & Poloczek, 2018, Sect. 3.4).

A cheaper alternative to the semidefinite relaxation: a random walk over {0, 1}^d that
flips one bit at a time, always accepts improvements, and accepts a worse point with
probability exp(delta / T) for a temperature T that cools down over the walk.
"""

from __future__ import annotations

import numpy as np

from ._model import quadratic_form


def _flip_gains(b: np.ndarray, A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Change of b @ x + x @ A @ x when each single bit of x is flipped.

    With A symmetric and zero on the diagonal, flipping bit i changes the objective by
    (1 - 2 x_i) * (b_i + 2 (A @ x)_i).
    """
    return (1.0 - 2.0 * x) * (b + 2.0 * (A @ x))


def maximize_sa(
    alpha: np.ndarray,
    n_vars: int,
    rng: np.random.Generator,
    *,
    penalty: float = 0.0,
    n_steps: int | None = None,
) -> np.ndarray:
    """Approximately maximize the sampled model minus penalty * sum(x) over {0, 1}^d.

    Parameters
    ----------
    alpha:
        One posterior sample of the second-order model coefficients.
    n_vars:
        Number of binary variables d.
    rng:
        NumPy random generator for the start point, the proposals and acceptances.
    penalty:
        Weight lambda of the L1 regularizer P(x) = sum(x) from the paper.
    n_steps:
        Number of proposed single-bit flips. Defaults to 10 * d^2, following the O(d^2)
        steps stated in the paper.

    Returns
    -------
    np.ndarray
        The best point visited, with shape (d,) and entries in {0, 1}.

    Notes
    -----
    The paper refers to the schedule of Spears (1993) without giving constants. Here the
    temperature cools geometrically from the mean absolute single-flip change at the
    start point down to 1/1000 of it.
    """
    _, b, A = quadratic_form(alpha, n_vars)
    b = b - penalty
    if n_steps is None:
        n_steps = 10 * n_vars**2

    x = rng.integers(0, 2, size=n_vars).astype(float)
    value = float(b @ x + x @ A @ x)
    best_x, best_value = x.copy(), value

    scale = float(np.mean(np.abs(_flip_gains(b, A, x))))
    if scale == 0.0:
        return best_x
    temperatures = scale * np.geomspace(1.0, 1e-3, n_steps)

    for temperature in temperatures:
        i = rng.integers(n_vars)
        gain = (1.0 - 2.0 * x[i]) * (b[i] + 2.0 * (A[i] @ x))
        if gain >= 0.0 or rng.random() < np.exp(gain / temperature):
            x[i] = 1.0 - x[i]
            value += gain
            if value > best_value:
                best_x, best_value = x.copy(), value
    return best_x
