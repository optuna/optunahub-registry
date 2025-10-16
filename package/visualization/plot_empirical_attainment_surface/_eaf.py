from __future__ import annotations

import numpy as np
import optuna


EPS = 1e-300


def _compute_emp_att_surf(
    X: np.ndarray, pareto_list: list[np.ndarray], levels: np.ndarray
) -> np.ndarray:
    """
    Compute the empirical attainment surface of the given Pareto front sets.

    Args:
        x:
            The first objective values appeared in pareto_list.
            This array is sorted in the ascending order.
            The shape is (number of possible values, ).
        levels:
            A list of `level` described below:
                Control the k in the k-% attainment surface.
                    k = level / n_independent_runs
                must hold.
                level must be in [1, n_independent_runs].
                level=1 leads to the best attainment surface,
                level=n_independent_runs leads to the worst attainment surface,
                level=n_independent_runs//2 leads to the median attainment surface.
        pareto_list:
            The list of the Pareto front sets.
            The shape is (trial number, Pareto solution index, objective index).
            Note that each pareto front set is sorted based on the ascending order of
            the first objective.

    Returns:
        emp_att_surfs (np.ndarray):
            The vertices of the empirical attainment surfaces for each level.
            If emp_att_surf[i, j, 1] takes np.inf, this is not actually on the surface.
            The shape is (levels.size, X.size, 2).

    Reference:
        Title: On the Computation of the Empirical Attainment Function
        Authors: Carlos M. Fonseca et al.
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.705.1929&rep=rep1&type=pdf

    NOTE:
        Our algorithm is slightly different from the original one, but the result will be same.
        More details below:
            When we define N = n_independent_runs, K = X.size, and S = n_samples,
            the original algorithm requires O(NK + K log K)
            and this algorithm requires O(NK log K).
            Although our algorithm is a bit worse than the original algorithm,
            since the enumerating Pareto solutions requires O(NS log S),
            which might be smaller complexity but will take more time in Python,
            the time complexity will not dominate the whole process.
    """
    n_levels = len(levels)
    emp_att_surfs = np.zeros((n_levels, X.size, 2))
    emp_att_surfs[..., 0] = X
    n_independent_runs = len(pareto_list)
    y_candidates = np.zeros((X.size, n_independent_runs))
    for i, pf_set in enumerate(pareto_list):
        ub = np.searchsorted(pf_set[:, 0], X, side="right")
        y_min = np.minimum.accumulate(np.hstack([np.inf, pf_set[:, 1]]))
        y_candidates[:, i] = y_min[ub]
    else:
        y_candidates = np.sort(y_candidates, axis=-1)

    y_sol = y_candidates[:, levels - 1].T
    emp_att_surfs[..., 1] = y_sol

    for emp_att_surf in emp_att_surfs:
        idx = np.sum(emp_att_surf[:, 1] == np.inf)
        emp_att_surf[:idx, 0] = emp_att_surf[idx, 0]

    return emp_att_surfs


def _get_empirical_attainment_surfaces(
    study_list: list[optuna.Study], levels: list[int], log_scale_inds: list[int] | None = None
) -> np.ndarray:
    """
    Get the empirical attainment surfaces given a list of studies.

    Args:
        study_list:
            A list of studies to visualize the empirical attainment function.
        levels:
            A list of `level` described below:
                Control the k in the k-% attainment surface.
                    k = level / n_independent_runs
                must hold.
                level must be in [1, n_independent_runs].
                level=1 leads to the best attainment surface,
                level=n_independent_runs leads to the worst attainment surface,
                level=n_independent_runs//2 leads to the median attainment surface.
        log_scale_inds:
            The indices of the log scale.
            For example, if you would like to plot the first objective in the log scale,
            you need to feed log_scale_inds=[0].
            In principle, log_scale_inds changes the minimum value of the axes
            from -np.inf to a small positive value.

    Returns:
        emp_att_surfs (np.ndarray):
            The surfaces attained by (level / n_independent_runs) * 100% (for each level in levels)
            of the trials. In other words, (level / n_independent_runs) * 100% of runs dominate or
            at least include those solutions in their Pareto front. Note that we only return the
            Pareto front of attained solutions.
    """
    if any(len(study.directions) != 2 for study in study_list):
        raise NotImplementedError("Three or more objectives are not supported.")
    if not all(1 <= level <= len(study_list) for level in levels):
        raise ValueError(
            f"All elements in levels must be in [1, {len(study_list)=}], but got {levels=}."
        )
    if not np.all(np.maximum.accumulate(levels) == levels):
        raise ValueError(f"levels must be an increasing sequence, but got {levels}.")
    larger_is_better_objectives = [
        i
        for i, d in enumerate(study_list[0].directions)
        if d == optuna.study.StudyDirection.MAXIMIZE
    ]
    pareto_list = []
    for study in study_list:
        trials = study._get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        sign_ = np.ones(2, dtype=float)
        sign_[larger_is_better_objectives] = -1
        loss_vals = sign_ * np.array([t.values for t in trials])
        on_front = optuna.study._multi_objective._is_pareto_front(
            loss_vals, assume_unique_lexsorted=False
        )
        pareto_list.append(loss_vals[on_front])

    log_scale_inds = log_scale_inds or []
    pareto_sols = np.vstack(pareto_list)
    X = np.unique(np.hstack([EPS if 0 in log_scale_inds else -np.inf, pareto_sols[:, 0], np.inf]))
    emp_att_surfs = _compute_emp_att_surf(X=X, pareto_list=pareto_list, levels=np.asarray(levels))
    emp_att_surfs[..., larger_is_better_objectives] *= -1
    if larger_is_better_objectives is not None and 0 in larger_is_better_objectives:
        emp_att_surfs = np.flip(emp_att_surfs, axis=1)

    return emp_att_surfs
