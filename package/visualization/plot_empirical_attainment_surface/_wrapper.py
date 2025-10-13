from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import optuna

from ._eaf import _get_empirical_attainment_surfaces
from ._visualizer import EmpiricalAttainmentFuncPlot


def plot_empirical_attainment_surface(
    study_list: list[optuna.Study],
    attainment_ratios: list[float],
    ax: plt.Axes | None = None,
    color: str | None = None,
    label: str | None = None,
    linestyle: str | None = None,
    marker: str | None = None,
    log_scale_inds: list[int] | None = None,
    **ax_plot_kwargs: Any,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()

    levels = [math.ceil(r * len(study_list)) for r in attainment_ratios]
    surfs = _get_empirical_attainment_surfaces(study_list, levels, log_scale_inds)
    plotter = EmpiricalAttainmentFuncPlot()
    if len(attainment_ratios) == 1:
        ax = plotter.plot_surface(
            ax,
            surfs[0],
            color=color,
            label=label,
            linestyle=linestyle,
            marker=marker,
            **ax_plot_kwargs,
        )
    elif len(attainment_ratios) == 3:
        plotter.plot_surface_with_band(
            ax,
            surfs,
            color=color,
            label=label,
            linestyle=linestyle,
            marker=marker,
            **ax_plot_kwargs,
        )
    else:
        raise ValueError("attainment_ratios must be a list of one or three elements.")

    return ax


def plot_multiple_empirical_attainment_surfaces(
    multiple_study_list: list[list[optuna.Study]],
    attainment_ratios: list[float],
    ax: plt.Axes | None = None,
    colors: list[str | None] | None = None,
    labels: list[str | None] | None = None,
    linestyles: list[str | None] | None = None,
    markers: list[str | None] | None = None,
    log_scale_inds: list[int] | None = None,
    **ax_plot_kwargs: Any,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()

    surfs_list = []
    for study_list in multiple_study_list:
        levels = [math.ceil(r * len(study_list)) for r in attainment_ratios]
        surfs_list.append(_get_empirical_attainment_surfaces(study_list, levels, log_scale_inds))

    plotter = EmpiricalAttainmentFuncPlot()
    if len(attainment_ratios) == 1:
        surfs_list = [surfs[0] for surfs in surfs_list]
        ax = plotter.plot_multiple_surface(
            ax,
            surfs_list,
            colors=colors,
            labels=labels,
            linestyles=linestyles,
            markers=markers,
            **ax_plot_kwargs,
        )
    elif len(attainment_ratios) == 3:
        plotter.plot_multiple_surface_with_band(
            ax,
            surfs_list,
            colors=colors,
            labels=labels,
            linestyles=linestyles,
            markers=markers,
            **ax_plot_kwargs,
        )
    else:
        raise ValueError("attainment_ratios must be a list of one or three elements.")

    ax.legend()
    return ax
