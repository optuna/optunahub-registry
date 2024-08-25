from __future__ import annotations

from collections.abc import Sequence
from optuna.logging import get_logger
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._pareto_front import (
    _get_pareto_front_info,
    _ParetoFrontInfo,
)
from optuna.visualization._utils import _make_hovertext
from optuna.visualization._plotly_imports import _imports


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


def plot_pareto_front(
    studies: Sequence[Study],
) -> "go.Figure":
    """Plot pareto front for each study.

    Args:
        studies:
            A list of study object whose trials are plotted for their pareto fronts.
            The number of objectives must be 2 or 3 for all trials and must be the same on all trials.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.
    """

    _imports.check()

    if not all(study._is_multi_objective() for study in studies):
        raise ValueError(
            "All studies must be multi-objective. For single-objective optimization, "
            "please use plot_optimization_history instead."
        )

    if not all(
        len(studies[0].directions) == len(study.directions) for study in studies
    ):
        raise ValueError("The number of objectives must be the same for all studies.")

    if not all(len(study.directions) in [2, 3] for study in studies):
        raise ValueError("The number of objectives must be 2 or 3 for all studies.")

    info_list: dict[str, _ParetoFrontInfo] = {}
    for study in studies:
        info = _get_pareto_front_info(study, include_dominated_trials=False)
        info_list[study.study_name] = info
    return _get_pareto_front_plot(info_list)


def _get_pareto_front_plot(
    info_dict: dict[str, _ParetoFrontInfo],
) -> "go.Figure":
    title = "Pareto-front Plot for multiple studies"
    info = info_dict[next(iter(info_dict))]
    if info.n_targets == 2:
        layout = go.Layout(
            title=title,
            xaxis_title=info.target_names[info.axis_order[0]],
            yaxis_title=info.target_names[info.axis_order[1]],
        )
    else:
        layout = go.Layout(
            title=title,
            scene={
                "xaxis_title": info.target_names[info.axis_order[0]],
                "yaxis_title": info.target_names[info.axis_order[1]],
                "zaxis_title": info.target_names[info.axis_order[2]],
            },
        )

    traces = []
    for key, value in info_dict.items():
        data = _make_scatter_object(
            key,
            value.n_targets,
            value.axis_order,
            value.best_trials_with_values,
        )
        traces.append(data)

    return go.Figure(data=traces, layout=layout)


def _make_scatter_object(
    study_name: str,
    n_targets: int,
    axis_order: Sequence[int],
    trials_with_values: Sequence[tuple[FrozenTrial, Sequence[float]]],
) -> "go.Scatter | go.Scatter3d":
    marker = {"line": {"width": 0.5, "color": "Grey"}}

    if n_targets == 2:
        return go.Scatter(
            name=study_name,
            x=[values[axis_order[0]] for _, values in trials_with_values],
            y=[values[axis_order[1]] for _, values in trials_with_values],
            text=[_make_hovertext(trial) for trial, _ in trials_with_values],
            mode="markers",
            marker=marker,
            showlegend=True,
        )
    elif n_targets == 3:
        return go.Scatter3d(
            name=study_name,
            x=[values[axis_order[0]] for _, values in trials_with_values],
            y=[values[axis_order[1]] for _, values in trials_with_values],
            z=[values[axis_order[2]] for _, values in trials_with_values],
            text=[_make_hovertext(trial) for trial, _ in trials_with_values],
            mode="markers",
            marker=marker,
            showlegend=True,
        )
    else:
        assert False, "Must not reach here"
