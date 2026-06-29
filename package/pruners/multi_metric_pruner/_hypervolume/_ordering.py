"""Full hypervolume-contribution ordering, layered on top of the HSSP selector.

This is kept separate from the vendored ``hssp.py`` (a faithful copy of Optuna's selector)
on purpose: ``_solve_hssp`` only contracts *which* indices survive, not their order. Once the
requested subset covers every (unique) point it short-circuits to original / lexicographic
order, so it cannot be used directly to *rank* a front. This module adds that ranking concern
without changing the copy.
"""

from __future__ import annotations

import numpy as np

from .hssp import _solve_hssp_on_unique_loss_vals


def _argsort_by_hv_contribution(loss_vals: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    """Order the rows of ``loss_vals`` best-first by greedy hypervolume contribution.

    Identical loss vectors have the same contribution, so they are treated as ties and emitted
    as one adjacent group (in ascending original-index order). The greedy order is computed on
    the unique rows and then expanded back to the original indices.

    Args:
        loss_vals: Loss values (lower is better) of shape ``(n_points, n_objectives)``.
        ref_point: A finite reference point that weakly dominates every row of ``loss_vals``.
            The resulting order depends on it.

    Returns:
        A permutation of ``range(len(loss_vals))``; the first index is the largest-contribution
        point and the last is the smallest.
    """
    assert np.isfinite(ref_point).all(), "ref_point must be finite to obtain a greedy order."
    n = len(loss_vals)
    if n <= 1:
        return np.arange(n)

    # `np.unique(axis=0)` returns lexsorted unique rows, which `_solve_hssp_on_unique_loss_vals`
    # (the 2D path in particular) assumes.
    unique_vals, inverse = np.unique(loss_vals, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    n_unique = len(unique_vals)
    if n_unique == 1:
        unique_order = np.zeros(1, dtype=int)
    else:
        unique_indices = np.arange(n_unique)
        # The selector short-circuits to non-greedy order once the subset covers every unique
        # point, so we request one fewer to stay on the greedy path; the single omitted point is
        # the smallest contributor and is appended last.
        greedy = _solve_hssp_on_unique_loss_vals(
            unique_vals, unique_indices, n_unique - 1, ref_point
        )
        leftover = np.setdiff1d(unique_indices, greedy)
        unique_order = np.concatenate([greedy, leftover])

    return np.concatenate([np.flatnonzero(inverse == u) for u in unique_order])
