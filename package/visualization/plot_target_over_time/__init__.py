from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.study import StudyDirection
from optuna.trial import TrialState


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D


def _get_values_on_fixed_time_steps(
    cumtime_list: list[np.ndarray],
    target_list: list[np.ndarray],
    log_time_scale: bool,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    t_min = np.min(np.stack(cumtime_list))
    t_max = np.max(np.stack(cumtime_list))
    if log_time_scale:
        ts = np.exp(np.linspace(np.log(t_min), np.log(t_max), num=n_steps))
    else:
        ts = np.linspace(t_min, t_max, num=n_steps)
    v_on_grid = []
    for ct, v in zip(cumtime_list, target_list):
        i_upper = np.minimum(np.searchsorted(ct, ts, side="left"), v.size - 1)
        v_on_grid.append(v[i_upper])
    return ts, np.array(v_on_grid)


def _validate(
    valid_states: tuple[TrialState, ...],
    states: tuple[TrialState, ...],
    target: Callable[[optuna.trial.FrozenTrial], float] | None,
    target_direction: StudyDirection | str | None,
) -> None:
    if any(s not in valid_states for s in states):
        raise ValueError(f"{states=} must be in {valid_states}.")
    if target_direction is None:
        if target is not None:
            raise ValueError("target was specified, but got target_direction=None.")
    else:
        if target is None:
            raise ValueError("target_direction was provided, but got target=None.")
        if target_direction not in [
            "minimize",
            "maximize",
            StudyDirection.MAXIMIZE,
            StudyDirection.MINIMIZE,
        ]:
            raise ValueError(
                f"target_direction must be either `minimize` or `maximize` but got {target_direction=}"
            )


def plot_target_over_time(
    study_list: list[optuna.Study],
    *,
    color: str,
    ax: Axes | None = None,
    states: tuple[TrialState, ...] | None = None,
    target: Callable[[optuna.trial.FrozenTrial], float] | None = None,
    target_direction: optuna.study.StudyDirection | str | None = None,
    cumtime_func: Callable[[optuna.trial.FrozenTrial], float] | None = None,
    log_time_scale: bool = True,
    n_steps: int = 100,
    **plot_kwargs: Any,
) -> Line2D:
    if ax is None:
        _, ax = plt.subplots()

    valid_states = (TrialState.COMPLETE, TrialState.PRUNED)
    states = states or valid_states
    assert states is not None, "Mypy Redefinition."
    _validate(valid_states, states, target, target_direction)

    target_list = []
    cumtime_list = []
    direction = target_direction or study_list[0].direction
    for study in study_list:
        trials = study.get_trials(deepcopy=False, states=states)
        target_vals = np.array([target(t) if target is not None else t.value for t in trials])
        if direction in ["minimize", StudyDirection.MINIMIZE]:
            target_list.append(np.minimum.accumulate(target_vals))
        else:
            target_list.append(np.maximum.accumulate(target_vals))
        if cumtime_func is not None:
            cumtime_list.append(np.array([cumtime_func(t) for t in trials]))
        else:
            datetime_start = min(t.datetime_start for t in trials if t.datetime_start is not None)
            cumtimes = np.array(
                [
                    (t.datetime_complete - datetime_start).total_seconds()
                    for t in trials
                    if t.datetime_complete is not None
                ]
            )
            cumtime_list.append(cumtimes)
        order = np.argsort(cumtime_list[-1])
        cumtime_list[-1] = cumtime_list[-1][order]
        if direction in ["minimize", StudyDirection.MINIMIZE]:
            target_list.append(np.minimum.accumulate(target_vals[order]))
        else:
            target_list.append(np.maximum.accumulate(target_vals[order]))

    ts, vs = _get_values_on_fixed_time_steps(
        cumtime_list,
        target_list,
        log_time_scale,
        n_steps,
    )
    m = np.mean(vs, axis=0)
    s = np.std(vs, axis=0) / np.sqrt(len(study_list))
    (line,) = ax.plot(ts, m, color=color, **plot_kwargs)
    ax.fill_between(ts, m - s, m + s, color=color, alpha=0.2)
    return line


__all__ = ["plot_target_over_time"]
