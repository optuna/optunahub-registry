from __future__ import annotations

import math
from typing import cast
from typing import TYPE_CHECKING

import numpy as np
from optuna._warnings import optuna_warn
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from optuna.study import Study


_CONSTRAINTS_KEY = "constraints"


def _process_constraints_after_trial(
    constraints_func: Callable[[FrozenTrial], Sequence[float]],
    study: Study,
    trial: FrozenTrial,
    state: TrialState,
) -> None:
    if state not in [TrialState.COMPLETE, TrialState.PRUNED]:
        return

    constraints = None
    try:
        con = constraints_func(trial)
        if np.any(np.isnan(con)):
            raise ValueError("Constraint values cannot be NaN.")
        if not isinstance(con, (tuple, list)):
            optuna_warn(
                f"Constraints should be a sequence of floats but got {type(con).__name__}."
            )
        constraints = tuple(con)
    finally:
        assert constraints is None or isinstance(constraints, tuple)

        study._storage.set_trial_system_attr(
            trial._trial_id,
            _CONSTRAINTS_KEY,
            constraints,
        )


def _is_pareto_front_nd(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    # NOTE(nabenabe0928): I tried the Kung's algorithm below, but it was not really quick.
    # https://github.com/optuna/optuna/pull/5302#issuecomment-1988665532
    # As unique_lexsorted_loss_values[:, 0] is sorted, we do not need it to judge dominance.
    loss_values = unique_lexsorted_loss_values[:, 1:]
    n_trials = loss_values.shape[0]
    on_front = np.zeros(n_trials, dtype=bool)
    remaining_indices: np.ndarray[tuple[int], np.dtype[np.signedinteger]] = np.arange(n_trials)
    # NOTE(nabenabe): Please check `_compute_exclusive_hv` in wfg.py when you modify this function.
    while len(remaining_indices):
        # NOTE: trials[j] cannot dominate trials[i] for i < j because of lexsort.
        # Therefore, remaining_indices[0] is always non-dominated.
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
    on_front[1:] = cummin_value1[1:] < cummin_value1[:-1]  # True if cummin value1 is new minimum.
    return on_front


def _is_pareto_front_for_unique_sorted(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    (n_trials, n_objectives) = unique_lexsorted_loss_values.shape
    if n_objectives == 1:
        on_front = np.zeros(len(unique_lexsorted_loss_values), dtype=bool)
        on_front[0] = True  # Only the first element is Pareto optimal.
        return on_front
    elif n_objectives == 2:
        return _is_pareto_front_2d(unique_lexsorted_loss_values)
    else:
        return _is_pareto_front_nd(unique_lexsorted_loss_values)


def _is_pareto_front(loss_values: np.ndarray, assume_unique_lexsorted: bool) -> np.ndarray:
    # NOTE(nabenabe): If assume_unique_lexsorted=True, but loss_values is not a unique array,
    # Duplicated Pareto solutions will be filtered out except for the earliest occurrences.
    # If assume_unique_lexsorted=True and loss_values[:, 0] is not sorted, then the result will be
    # incorrect.
    if assume_unique_lexsorted:
        return _is_pareto_front_for_unique_sorted(loss_values)

    unique_lexsorted_loss_values, order_inv = np.unique(loss_values, axis=0, return_inverse=True)
    on_front = _is_pareto_front_for_unique_sorted(unique_lexsorted_loss_values)
    # NOTE(nabenabe): We can remove `.reshape(-1)` if ``numpy==2.0.0`` is not used.
    # https://github.com/numpy/numpy/issues/26738
    # TODO: Remove `.reshape(-1)` once `numpy==2.0.0` is obsolete.
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

    # It ensures that trials[j] will not dominate trials[i] for i < j.
    # np.unique does lexsort.
    unique_lexsorted_loss_values, order_inv = np.unique(loss_values, return_inverse=True, axis=0)
    n_unique = unique_lexsorted_loss_values.shape[0]
    # Clip n_below.
    n_below = min(n_below or len(unique_lexsorted_loss_values), len(unique_lexsorted_loss_values))
    ranks = np.zeros(n_unique, dtype=int)
    rank = 0
    indices = np.arange(n_unique)
    while n_unique - indices.size < n_below:
        on_front = _is_pareto_front(unique_lexsorted_loss_values, assume_unique_lexsorted=True)
        ranks[indices[on_front]] = rank
        # Remove the recent Pareto solutions.
        indices = indices[~on_front]
        unique_lexsorted_loss_values = unique_lexsorted_loss_values[~on_front]
        rank += 1

    ranks[indices] = rank  # Rank worse than the top n_below is defined as the worst rank.
    # NOTE(nabenabe): We can remove `.reshape(-1)` if ``numpy==2.0.0`` is not used.
    # https://github.com/numpy/numpy/issues/26738
    # TODO: Remove `.reshape(-1)` once `numpy==2.0.0` is obsolete.
    return ranks[order_inv.reshape(-1)]


def _fast_non_domination_rank(
    loss_values: np.ndarray, *, penalty: np.ndarray | None = None, n_below: int | None = None
) -> np.ndarray:
    """Calculate non-domination rank based on the fast non-dominated sort algorithm.

    The fast non-dominated sort algorithm assigns a rank to each trial based on the dominance
    relationship of the trials, determined by the objective values and the penalty values. The
    algorithm is based on `the constrained NSGA-II algorithm
    <https://doi.org/10.1109/4235.99601>`__, but the handling of the case when penalty
    values are None is different. The algorithm assigns the rank according to the following
    rules:

    1. Feasible trials: First, the algorithm assigns the rank to feasible trials, whose penalty
        values are less than or equal to 0, according to unconstrained version of fast non-
        dominated sort.
    2. Infeasible trials: Next, the algorithm assigns the rank from the minimum penalty value of to
        the maximum penalty value.
    3. Trials with no penalty information (constraints value is None): Finally, The algorithm
        assigns the rank to trials with no penalty information according to unconstrained version
        of fast non-dominated sort. Note that only this step is different from the original
        constrained NSGA-II algorithm.
    Plus, the algorithm terminates whenever the number of sorted trials reaches n_below.

    Args:
        loss_values:
            Objective values, which is better when it is lower, of each trials.
        penalty:
            Constraints values of each trials. Defaults to None.
        n_below: The minimum number of top trials required to be sorted. The algorithm will
            terminate when the number of sorted trials reaches n_below. Defaults to None.

    Returns:
        An ndarray in the shape of (n_trials,), where each element is the non-domination rank of
        each trial. The rank is 0-indexed. This function guarantees the correctness of the ranks
        only up to the top-``n_below`` solutions. If a solution's rank is worse than the
        top-``n_below`` solution, its rank will be guaranteed to be greater than the rank of
        the top-``n_below`` solution.
    """
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

    # First, we calculate the domination rank for feasible trials.
    ranks[is_feasible] = _calculate_nondomination_rank(loss_values[is_feasible], n_below=n_below)
    n_below -= int(np.count_nonzero(is_feasible))

    # Second, we calculate the domination rank for infeasible trials.
    top_rank_infeasible = np.max(ranks[is_feasible], initial=-1) + 1
    ranks[is_infeasible] = top_rank_infeasible + _calculate_nondomination_rank(
        penalty[is_infeasible][:, np.newaxis], n_below=n_below
    )
    n_below -= int(np.count_nonzero(is_infeasible))

    # Third, we calculate the domination rank for trials with no penalty information.
    top_rank_penalty_nan = np.max(ranks[~is_penalty_nan], initial=-1) + 1
    ranks[is_penalty_nan] = top_rank_penalty_nan + _calculate_nondomination_rank(
        loss_values[is_penalty_nan], n_below=n_below
    )
    assert np.all(ranks != -1), "All the rank must be updated."
    return ranks


def _compute_2d(sorted_pareto_sols: np.ndarray, reference_point: np.ndarray) -> float:
    assert sorted_pareto_sols.shape[1] == reference_point.shape[0] == 2
    rect_diag_y = np.concatenate([reference_point[1:], sorted_pareto_sols[:-1, 1]])
    edge_length_x = reference_point[0] - sorted_pareto_sols[:, 0]
    edge_length_y = rect_diag_y - sorted_pareto_sols[:, 1]
    return edge_length_x @ edge_length_y


def _compute_3d(sorted_pareto_sols: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Compute hypervolume in 3D. Time complexity is O(N^2) where N is sorted_pareto_sols.shape[0].
    If X, Y, Z coordinates are permutations of 0, 1, ..., N-1 and reference_point is (N, N, N), the
    hypervolume is calculated as the number of voxels (x, y, z) dominated by at least one point.
    If we fix x and y, this number is equal to the minimum of z' over all points (x', y', z')
    satisfying x' <= x and y' <= y. This can be efficiently computed using cumulative minimum
    (`np.minimum.accumulate`). Non-permutation coordinates can be transformed into permutation
    coordinates by using coordinate compression.
    """
    assert sorted_pareto_sols.shape[1] == reference_point.shape[0] == 3
    n = sorted_pareto_sols.shape[0]
    y_order = np.argsort(sorted_pareto_sols[:, 1])
    z_delta = np.zeros((n, n), dtype=float)
    z_delta[y_order, np.arange(n)] = reference_point[2] - sorted_pareto_sols[y_order, 2]
    z_delta = np.maximum.accumulate(np.maximum.accumulate(z_delta, axis=0), axis=1)
    # The x axis is already sorted, so no need to compress this coordinate.
    x_vals = sorted_pareto_sols[:, 0]
    y_vals = sorted_pareto_sols[y_order, 1]
    x_delta = np.concatenate([x_vals[1:], reference_point[:1]]) - x_vals
    y_delta = np.concatenate([y_vals[1:], reference_point[1:2]]) - y_vals
    # NOTE(nabenabe): Below is the faster alternative of `np.sum(dx[:, None] * dy * dz)`.
    return np.dot(np.dot(z_delta, y_delta), x_delta)


def _compute_hv(sorted_loss_vals: np.ndarray, reference_point: np.ndarray) -> float:
    if sorted_loss_vals.shape[0] == 1:
        # NOTE(nabenabe): NumPy overhead is slower than the simple for-loop here.
        inclusive_hv = 1.0
        for r, v in zip(reference_point, sorted_loss_vals[0]):
            inclusive_hv *= r - v
        return float(inclusive_hv)
    elif sorted_loss_vals.shape[0] == 2:
        # NOTE(nabenabe): NumPy overhead is slower than the simple for-loop here.
        # S(A v B) = S(A) + S(B) - S(A ^ B).
        hv1, hv2, intersec = 1.0, 1.0, 1.0
        for r, v1, v2 in zip(reference_point, sorted_loss_vals[0], sorted_loss_vals[1]):
            hv1 *= r - v1
            hv2 *= r - v2
            intersec *= r - max(v1, v2)
        return hv1 + hv2 - intersec

    inclusive_hvs = (reference_point - sorted_loss_vals).prod(axis=-1)
    # c.f. Eqs. (6) and (7) of ``A Fast Way of Calculating Exact Hypervolumes``.
    limited_sols_array = np.maximum(sorted_loss_vals[:, np.newaxis], sorted_loss_vals)
    return inclusive_hvs[-1] + sum(
        _compute_exclusive_hv(limited_sols_array[i, i + 1 :], inclusive_hvs[i], reference_point)
        for i in range(inclusive_hvs.size - 1)
    )


def _compute_exclusive_hv(
    limited_sols: np.ndarray, inclusive_hv: float, reference_point: np.ndarray
) -> float:
    assert limited_sols.shape[0] >= 1
    if limited_sols.shape[0] <= 3:
        # NOTE(nabenabe): Don't use _is_pareto_front for 3 or fewer points to avoid its overhead.
        return inclusive_hv - _compute_hv(limited_sols, reference_point)

    # NOTE(nabenabe): The following holds only for an incremental version of `_is_pareto_front_nd`,
    # meaning that if there are duplicated Pareto solutions, the second one must be judged as
    # weakly dominated by the first one.
    # NOTE(nabenabe): As the following line is a hack for speedup, I will describe several
    # important points to note. Even if we do not run _is_pareto_front below or use
    # assume_unique_lexsorted=False instead, the result of this function does not change, but this
    # function simply becomes slower.
    #
    # For simplicity, I call an array ``quasi-lexsorted`` if it is sorted by the first objective.
    #
    # Reason why it will be faster with _is_pareto_front
    #   Hypervolume of a given solution set and a reference point does not change even when we
    #   remove non Pareto solutions from the solution set. However, the calculation becomes slower
    #   if the solution set contains many non Pareto solutions. By removing some obvious non Pareto
    #   solutions, the calculation becomes faster.
    #
    # Reason why assume_unique_lexsorted must be True for _is_pareto_front
    #   assume_unique_lexsorted=True actually checks weak dominance and solutions will be weakly
    #   dominated if there are duplications, so we can remove duplicated solutions by this option.
    #   In other words, assume_unique_lexsorted=False may significantly slow down when limited_sols
    #   has many duplicated Pareto solutions because this function becomes an exponential algorithm
    #   without duplication removal.
    #
    # NOTE(nabenabe): limited_sols can be non-unique and/or non-lexsorted, so I will describe why
    # it is fine.
    #
    # Reason why we can specify assume_unique_lexsorted=True even when limited_sols is not
    #   All ``False`` in on_front will be correct (, but it may not be the case for ``True``) even
    #   if limited_sols is not unique or not lexsorted as long as limited_sols is quasi-lexsorted,
    #   which is guaranteed. As mentioned earlier, if all ``False`` in on_front is correct, the
    #   result of this function does not change.
    on_front = _is_pareto_front(limited_sols, assume_unique_lexsorted=True)
    return inclusive_hv - _compute_hv(limited_sols[on_front], reference_point)


def compute_hypervolume(
    loss_vals: np.ndarray, reference_point: np.ndarray, assume_pareto: bool = False
) -> float:
    """Hypervolume calculator for any dimension.

    This class exactly calculates the hypervolume for any dimension.
    For 3 dimensions or higher, the WFG algorithm will be used.
    Please refer to ``A Fast Way of Calculating Exact Hypervolumes`` for the WFG algorithm.

    .. note::
        This class is used for computing the hypervolumes of points in multi-objective space.
        Each coordinate of each point represents a ``values`` of the multi-objective function.

    .. note::
        We check that each objective is to be minimized. Transform objective values that are
        to be maximized before calling this class's ``compute`` method.

    Args:
        loss_vals:
            An array of loss value vectors to calculate the hypervolume.
        reference_point:
            The reference point used to calculate the hypervolume.
        assume_pareto:
            Whether to assume the Pareto optimality to ``loss_vals``.
            In other words, if ``True``, none of loss vectors are dominated by another.
            ``assume_pareto`` is used only for speedup and it does not change the result even if
            this argument is wrongly given. If there are many non-Pareto solutions in
            ``loss_vals``, ``assume_pareto=True`` will speed up the calculation.

    Returns:
        The hypervolume of the given arguments.

    """

    if not np.all(loss_vals <= reference_point):
        raise ValueError(
            "All points must dominate or equal the reference point. "
            "That is, for all points in the loss_vals and the coordinate `i`, "
            "`loss_vals[i] <= reference_point[i]`."
        )
    if not np.all(np.isfinite(reference_point)):
        # reference_point does not have nan, thanks to the verification above.
        return float("inf")
    if loss_vals.size == 0:
        return 0.0

    if not assume_pareto:
        unique_lexsorted_loss_vals = np.unique(loss_vals, axis=0)
        on_front = _is_pareto_front(unique_lexsorted_loss_vals, assume_unique_lexsorted=True)
        sorted_pareto_sols = unique_lexsorted_loss_vals[on_front]
    else:
        # NOTE(nabenabe): The result of this function does not change both by
        # np.argsort(loss_vals[:, 0]) and np.unique(loss_vals, axis=0).
        # But many duplications in loss_vals significantly slows down the function.
        # TODO(nabenabe): Make an option to use np.unique.
        sorted_pareto_sols = loss_vals[loss_vals[:, 0].argsort()]

    if reference_point.shape[0] == 2:
        hv = _compute_2d(sorted_pareto_sols, reference_point)
    elif reference_point.shape[0] == 3:
        # NOTE: For 3D points, we always prefer _compute_3d to _compute_hv because the time
        # complexity of _compute_3d is O(N^2), while that of _compute_nd is \\Omega(N^3)
        # - It calls _compute_exclusive_hv with i points for i = 0, 1, ..., N-1
        # - _compute_exclusive_hv calls _is_pareto_front, which is quadratic
        #   with the number of points
        hv = _compute_3d(sorted_pareto_sols, reference_point)
    else:
        hv = _compute_hv(sorted_pareto_sols, reference_point)

    # NOTE(nabenabe): `nan` happens when inf - inf happens, but this is inf in hypervolume due to
    # the submodularity.
    return hv if np.isfinite(hv) else float("inf")


def _solve_hssp_2d(
    rank_i_loss_vals: np.ndarray,
    rank_i_indices: np.ndarray,
    subset_size: int,
    reference_point: np.ndarray,
) -> np.ndarray:
    # This function can be used for non-unique rank_i_loss_vals as well.
    # The time complexity is O(subset_size * rank_i_loss_vals.shape[0]).
    assert rank_i_loss_vals.shape[-1] == 2 and subset_size <= rank_i_loss_vals.shape[0]
    n_trials = rank_i_loss_vals.shape[0]
    # rank_i_loss_vals is unique-lexsorted in solve_hssp.
    sorted_indices = np.arange(rank_i_loss_vals.shape[0])
    sorted_loss_vals = rank_i_loss_vals.copy()
    # The diagonal points for each rectangular to calculate the hypervolume contributions.
    rect_diags = np.repeat(reference_point[np.newaxis, :], n_trials, axis=0)
    selected_indices = np.zeros(subset_size, dtype=int)
    for i in range(subset_size):
        contribs = np.prod(rect_diags - sorted_loss_vals, axis=-1)
        max_index = np.argmax(contribs)
        selected_indices[i] = rank_i_indices[sorted_indices[max_index]]
        loss_vals = sorted_loss_vals[max_index].copy()

        keep = np.ones(n_trials - i, dtype=bool)
        keep[max_index] = False
        # Remove the chosen point.
        sorted_indices = sorted_indices[keep]
        rect_diags = rect_diags[keep]
        sorted_loss_vals = sorted_loss_vals[keep]
        # Update the diagonal points for each hypervolume contribution calculation.
        rect_diags[:max_index, 0] = np.minimum(loss_vals[0], rect_diags[:max_index, 0])
        rect_diags[max_index:, 1] = np.minimum(loss_vals[1], rect_diags[max_index:, 1])

    return selected_indices


def _lazy_contribs_update(
    contribs: np.ndarray,
    pareto_loss_values: np.ndarray,
    selected_vecs: np.ndarray,
    reference_point: np.ndarray,
    hv_selected: float,
) -> np.ndarray:
    """Lazy update the hypervolume contributions.

    (1) Lazy update of the hypervolume contributions
    S=selected_indices - {indices[max_index]}, T=selected_indices, and S' is a subset of S.
    As we would like to know argmax H(T v {i}) in the next iteration, we can skip HV
    calculations for j if H(T v {i}) - H(T) > H(S' v {j}) - H(S') >= H(T v {j}) - H(T).
    We used the submodularity for the inequality above. As the upper bound of contribs[i] is
    H(S' v {j}) - H(S'), we start to update from i with a higher upper bound so that we can
    skip more HV calculations.

    (2) A simple cheap-to-evaluate contribution upper bound
    The HV difference only using the latest selected point and a candidate is a simple, yet
    obvious, contribution upper bound. Denote t as the latest selected index and j as an unselected
    index. Then, H(T v {j}) - H(T) <= H({t} v {j}) - H({t}) holds where the inequality comes from
    submodularity. We use the inclusion-exclusion principle to calculate the RHS.
    """
    if math.isinf(hv_selected):
        # NOTE(nabenabe): This part eliminates the possibility of inf - inf in this function.
        return np.full_like(contribs, np.inf)

    intersec = np.maximum(pareto_loss_values[:, np.newaxis], selected_vecs[:-1])
    inclusive_hvs = np.prod(reference_point - pareto_loss_values, axis=1)
    is_contrib_inf = np.isinf(inclusive_hvs)  # NOTE(nabe): inclusive_hvs[i] >= contribs[i].
    contribs = np.minimum(  # Please see (2) in the docstring for more details.
        contribs, inclusive_hvs - np.prod(reference_point - intersec[:, -1], axis=1)
    )
    max_contrib = 0.0
    is_hv_calc_fast = pareto_loss_values.shape[1] <= 3
    for i in np.argsort(-contribs):  # Check from larger upper bound contribs to skip more.
        if is_contrib_inf[i]:
            max_contrib = contribs[i] = np.inf
            continue
        if contribs[i] < max_contrib:  # Please see (1) in the docstring for more details.
            continue

        # NOTE(nabenabe): contribs[i] = H(S v {i)) - H(S) = H({i}) - H(S ^ {i}).
        # If HV calc is fast, the decremental approach, which involves Pareto checks, is slower.
        if is_hv_calc_fast:  # Use contribs[i] = H(S v {i)) - H(S) (incremental approach).
            selected_vecs[-1] = pareto_loss_values[i].copy()
            hv_plus = compute_hypervolume(selected_vecs, reference_point, assume_pareto=True)
            contribs[i] = hv_plus - hv_selected
        else:  # Use contribs[i] = H({i}) - H(S ^ {i}) (decremental approach).
            contribs[i] = inclusive_hvs[i] - compute_hypervolume(intersec[i], reference_point)
        max_contrib = max(contribs[i], max_contrib)

    return contribs


def _solve_hssp_on_unique_loss_vals(
    rank_i_loss_vals: np.ndarray,
    rank_i_indices: np.ndarray,
    subset_size: int,
    reference_point: np.ndarray,
) -> np.ndarray:
    if not np.isfinite(reference_point).all():
        return rank_i_indices[:subset_size]
    if rank_i_indices.size == subset_size:
        return rank_i_indices
    if rank_i_loss_vals.shape[-1] == 2:
        return _solve_hssp_2d(rank_i_loss_vals, rank_i_indices, subset_size, reference_point)

    assert subset_size < rank_i_indices.size
    # The following logic can be used for non-unique rank_i_loss_vals as well.
    diff_of_loss_vals_and_ref_point = reference_point - rank_i_loss_vals
    (n_solutions, n_objectives) = rank_i_loss_vals.shape
    contribs = np.prod(diff_of_loss_vals_and_ref_point, axis=-1)
    selected_indices = np.zeros(subset_size, dtype=int)
    selected_vecs = np.empty((subset_size, n_objectives))
    indices = np.arange(n_solutions)
    hv = 0
    for k in range(subset_size):
        max_index = int(np.argmax(contribs))
        hv += contribs[max_index]
        selected_indices[k] = indices[max_index]
        selected_vecs[k] = rank_i_loss_vals[max_index].copy()
        keep = np.ones(contribs.size, dtype=bool)
        keep[max_index] = False
        contribs = contribs[keep]
        indices = indices[keep]
        rank_i_loss_vals = rank_i_loss_vals[keep]
        if k == subset_size - 1:
            # We do not need to update contribs at the last iteration.
            break

        contribs = _lazy_contribs_update(
            contribs, rank_i_loss_vals, selected_vecs[: k + 2], reference_point, hv
        )

    return rank_i_indices[selected_indices]


def _solve_hssp(
    rank_i_loss_vals: np.ndarray,
    rank_i_indices: np.ndarray,
    subset_size: int,
    reference_point: np.ndarray,
) -> np.ndarray:
    """Solve a hypervolume subset selection problem (HSSP) via a greedy algorithm.

    This method is a 1-1/e approximation algorithm to solve HSSP.

    For further information about algorithms to solve HSSP, please refer to the following
    paper:

    - `Greedy Hypervolume Subset Selection in Low Dimensions
       <https://doi.org/10.1162/EVCO_a_00188>`__
    """
    if subset_size == rank_i_indices.size:
        return rank_i_indices

    rank_i_unique_loss_vals, indices_of_unique_loss_vals = np.unique(
        rank_i_loss_vals, return_index=True, axis=0
    )
    n_unique = indices_of_unique_loss_vals.size
    if n_unique < subset_size:
        chosen = np.zeros(rank_i_indices.size, dtype=bool)
        chosen[indices_of_unique_loss_vals] = True
        duplicated_indices = np.arange(rank_i_indices.size)[~chosen]
        chosen[duplicated_indices[: subset_size - n_unique]] = True
        return rank_i_indices[chosen]

    selected_indices_of_unique_loss_vals = _solve_hssp_on_unique_loss_vals(
        rank_i_unique_loss_vals, indices_of_unique_loss_vals, subset_size, reference_point
    )
    return rank_i_indices[selected_indices_of_unique_loss_vals]
