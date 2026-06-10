from __future__ import annotations

import numpy as np
import optuna
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study import StudyDirection

from ._hv import compute_hypervolume


def get_hypervolume_history(
    study: optuna.Study, ref_point: np.ndarray | None = None
) -> np.ndarray:
    completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    if len(completed_trials) == 0:
        return np.empty(0, dtype=float)

    values_array = np.array([t.values for t in completed_trials])
    if _CONSTRAINTS_KEY in completed_trials[0].system_attrs:
        constraints = np.array([t.system_attrs(_CONSTRAINTS_KEY) for t in completed_trials])
        is_feasible = np.all(constraints <= 0.0, axis=-1)
    else:
        is_feasible = np.ones(len(completed_trials), dtype=bool)
    signs = np.asarray([1 if d == StudyDirection.MINIMIZE else -1 for d in study.directions])
    values_array *= signs
    if ref_point is not None:
        ref_point *= signs
    else:
        max_values = np.max(values_array, axis=0)
        ref_point = np.maximum(0.9 * max_values, 1.1 * max_values)
        ref_point[ref_point == 0] = 1e-12

    hh = np.zeros(len(completed_trials), dtype=float)
    pareto_sols: np.ndarray | None = None
    for i, (feas, values) in enumerate(zip(is_feasible, values_array)):
        hh[i] = hh[i - 1].item() if i > 0 else 0.0
        if not feas:
            continue

        new_values = values[np.newaxis]
        if pareto_sols is not None:
            if (pareto_sols <= new_values).all(axis=1).any(axis=0):
                continue
        if (new_values > ref_point).any():
            continue
        hh[i] += np.prod(ref_point - new_values)
        if pareto_sols is None:
            pareto_sols = new_values.copy()
        else:
            limited_sols = np.maximum(pareto_sols, new_values)
            hh[i] -= compute_hypervolume(limited_sols, ref_point)
            is_kept = (pareto_sols < new_values).any(axis=1)
            pareto_sols = np.concatenate([pareto_sols[is_kept, :], new_values], axis=0)

    return hh
