from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from optuna.logging import get_logger
from optuna.study import Study
from optuna.visualization._hypervolume_history import _get_hypervolume_history_info
from optuna.visualization._plotly_imports import _imports


class _HypervolumeHistoryInfo(NamedTuple):
    trial_numbers: list[int]
    values: list[float]
    study_name: str


if _imports.is_successful():
    from optuna.visualization._plotly_imports import go

_logger = get_logger(__name__)


def plot_hypervolume_history(
    studies: Sequence[Study],
    reference_point: Sequence[float],
) -> "go.Figure":
    """Plot hypervolume history for each study.

    Args:
        studies:
            A list of study object whose trials are plotted for their hypervolumes.
            The number of objectives must be 2 or more for all trials and must be the same on all trials.

        reference_point:
            A reference point to use for hypervolume computation.
            The dimension of the reference point must be the same as the number of objectives.

    Returns:
        A :class:`plotly.graph_objects.Figure` object.
    """

    _imports.check()

    if not all(study._is_multi_objective() for study in studies):
        raise ValueError(
            "All studies must be multi-objective. For single-objective optimization, "
            "please use plot_optimization_history instead."
        )

    if not all(len(study.directions) == len(study.directions) for study in studies):
        raise ValueError("The number of objectives must be the same for all studies.")

    if len(reference_point) != len(studies[0].directions):
        raise ValueError(
            "The dimension of the reference point must be the same as the number of objectives."
        )

    info_list = []
    for study in studies:
        info_ = _get_hypervolume_history_info(study, np.asarray(reference_point, dtype=np.float64))
        info_list.append(
            _HypervolumeHistoryInfo(info_.trial_numbers, info_.values, study.study_name)
        )
    return _get_hypervolume_history_plot(info_list)


def _get_hypervolume_history_plot(
    info_list: list[_HypervolumeHistoryInfo],
) -> "go.Figure":
    layout = go.Layout(
        title="Hypervolume History Plot",
        xaxis={"title": "Trial"},
        yaxis={"title": "Hypervolume"},
    )

    traces = []
    for info in info_list:
        data = go.Scatter(
            name=info.study_name,
            x=info.trial_numbers,
            y=info.values,
            mode="lines+markers",
        )
        traces.append(data)
    return go.Figure(data=traces, layout=layout)
