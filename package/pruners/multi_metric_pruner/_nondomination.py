"""Non-domination rank — copied from optuna/optuna/study/_multi_objective.py."""
from __future__ import annotations

from typing import cast

import numpy as np


def _is_pareto_front_nd(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    loss_values = unique_lexsorted_loss_values[:, 1:]
    n_trials = loss_values.shape[0]
    on_front = np.zeros(n_trials, dtype=bool)
    remaining_indices: np.ndarray[tuple[int], np.dtype[np.signedinteger]] = np.arange(n_trials)
    while len(remaining_indices):
        on_front[(new_nondominated_index := remaining_indices[0])] = True
        nondominated_and_not_top = np.any(
            loss_values[remaining_indices] < loss_values[new_nondominated_index], axis=1
        )
        remaining_indices = cast(
            "np.ndarray[tuple[int], np.dtype[np.signedinteger]]",
            remaining_indices[nondominated_and_not_top],
        )
    return on_front


def _is_pareto_front_2d(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    n_trials = unique_lexsorted_loss_values.shape[0]
    cummin_value1 = np.minimum.accumulate(unique_lexsorted_loss_values[:, 1])
    on_front = np.ones(n_trials, dtype=bool)
    on_front[1:] = cummin_value1[1:] < cummin_value1[:-1]
    return on_front


def _is_pareto_front_for_unique_sorted(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    (n_trials, n_objectives) = unique_lexsorted_loss_values.shape
    if n_objectives == 1:
        on_front = np.zeros(len(unique_lexsorted_loss_values), dtype=bool)
        on_front[0] = True
        return on_front
    elif n_objectives == 2:
        return _is_pareto_front_2d(unique_lexsorted_loss_values)
    else:
        return _is_pareto_front_nd(unique_lexsorted_loss_values)


def _is_pareto_front(loss_values: np.ndarray, assume_unique_lexsorted: bool) -> np.ndarray:
    if assume_unique_lexsorted:
        return _is_pareto_front_for_unique_sorted(loss_values)
    unique_lexsorted_loss_values, order_inv = np.unique(loss_values, axis=0, return_inverse=True)
    on_front = _is_pareto_front_for_unique_sorted(unique_lexsorted_loss_values)
    return on_front[order_inv.reshape(-1)]


def _calculate_nondomination_rank(
    loss_values: np.ndarray, *, n_below: int | None = None
) -> np.ndarray:
    if len(loss_values) == 0 or (n_below is not None and n_below <= 0):
        return np.zeros(len(loss_values), dtype=int)

    (n_trials, n_objectives) = loss_values.shape
    if n_objectives == 1:
        _, ranks = np.unique(loss_values[:, 0], return_inverse=True)
        return ranks

    unique_lexsorted_loss_values, order_inv = np.unique(loss_values, return_inverse=True, axis=0)
    n_unique = unique_lexsorted_loss_values.shape[0]
    n_below = min(n_below or len(unique_lexsorted_loss_values), len(unique_lexsorted_loss_values))
    ranks = np.zeros(n_unique, dtype=int)
    rank = 0
    indices = np.arange(n_unique)
    while n_unique - indices.size < n_below:
        on_front = _is_pareto_front(unique_lexsorted_loss_values, assume_unique_lexsorted=True)
        ranks[indices[on_front]] = rank
        indices = indices[~on_front]
        unique_lexsorted_loss_values = unique_lexsorted_loss_values[~on_front]
        rank += 1

    ranks[indices] = rank
    return ranks[order_inv.reshape(-1)]


def _fast_non_domination_rank(
    loss_values: np.ndarray, *, penalty: np.ndarray | None = None, n_below: int | None = None
) -> np.ndarray:
    if len(loss_values) == 0:
        return np.array([], dtype=int)

    n_below = n_below or len(loss_values)
    assert n_below > 0, "n_below must be a positive integer."

    if penalty is None:
        return _calculate_nondomination_rank(loss_values, n_below=n_below)

    if len(penalty) != len(loss_values):
        raise ValueError(
            "The length of penalty and loss_values must be same, but got "
            f"{len(penalty)=} and {len(loss_values)=}."
        )

    ranks = np.full(len(loss_values), -1, dtype=int)
    is_penalty_nan = np.isnan(penalty)
    is_feasible = np.logical_and(~is_penalty_nan, penalty <= 0)
    is_infeasible = np.logical_and(~is_penalty_nan, penalty > 0)

    ranks[is_feasible] = _calculate_nondomination_rank(loss_values[is_feasible], n_below=n_below)
    n_below -= int(np.count_nonzero(is_feasible))

    top_rank_infeasible = np.max(ranks[is_feasible], initial=-1) + 1
    ranks[is_infeasible] = top_rank_infeasible + _calculate_nondomination_rank(
        penalty[is_infeasible][:, np.newaxis], n_below=n_below
    )
    n_below -= int(np.count_nonzero(is_infeasible))

    top_rank_penalty_nan = np.max(ranks[~is_penalty_nan], initial=-1) + 1
    ranks[is_penalty_nan] = top_rank_penalty_nan + _calculate_nondomination_rank(
        loss_values[is_penalty_nan], n_below=n_below
    )
    assert np.all(ranks != -1), "All the rank must be updated."
    return ranks
