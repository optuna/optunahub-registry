from __future__ import annotations

from collections.abc import Callable
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

import numpy as np
from optuna.study import Study
from optuna.trial import FrozenTrial

from .._data import _get_plot_info
from .._data import _PlotInfo
from .._data import DEFAULT_MISSING_LABEL


if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_parallel_coordinate(
    study: Study,
    params: Optional[list[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
    objective_names: Optional[list[str]] = None,
    missing_label: str = DEFAULT_MISSING_LABEL,
) -> "Axes":
    """Plot a parallel coordinate plot with Matplotlib.

    Args:
        study:
            A study whose completed trials are plotted.
        params:
            Parameter names to visualize. If omitted, all parameters observed in completed trials
            are used.
        target:
            A target function. If specified, the plot behaves as a single-target plot even for
            multi-objective studies.
        target_name:
            Name of the target axis and colorbar when ``target`` is used, or for single-objective
            studies.
        objective_names:
            Axis labels for multi-objective values. If omitted, ``Objective 0``, ``Objective 1``,
            ... are used.
        missing_label:
            Tick label used for parameters missing from a trial.

    Returns:
        A :class:`matplotlib.axes.Axes` object.
    """

    info = _get_plot_info(study, params, target, target_name, objective_names, missing_label)
    return _get_parallel_coordinate_plot(info)


def _get_parallel_coordinate_plot(info: _PlotInfo) -> "Axes":
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_title("Parallel Coordinate Plot")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_visible(False)

    n_dimensions = len(info.dimensions)
    if n_dimensions == 0 or len(info.color_values) == 0:
        return ax

    if n_dimensions == 1:
        ax.set_xlim(-0.5, 0.5)
    else:
        ax.set_xlim(0, n_dimensions - 1)
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(range(n_dimensions))
    ax.set_xticklabels([dim.label for dim in info.dimensions], rotation=330, ha="left")
    ax.tick_params(axis="x", length=0)

    rows = np.asarray(
        [
            [dimension.values[trial_id] for dimension in info.dimensions]
            for trial_id in info.draw_order
        ],
        dtype=np.float64,
    )
    color_values = np.asarray([info.color_values[trial_id] for trial_id in info.draw_order])
    constraints_satisfied = np.asarray(
        [info.constraints_satisfied[trial_id] for trial_id in info.draw_order], dtype=bool
    )

    cmap = plt.get_cmap("Blues_r" if info.reverse_scale else "Blues")
    norm: Any
    colorbar_ticks: Optional[list[int]]
    if info.is_rank_color:
        max_rank = int(max(info.color_values))
        bounds = np.arange(-0.5, max_rank + 1.5, 1.0)
        norm = BoundaryNorm(bounds, cmap.N)
        colorbar_ticks = list(range(max_rank + 1))
    else:
        min_color = float(np.min(color_values))
        max_color = float(np.max(color_values))
        if min_color == max_color:
            min_color -= 0.5
            max_color += 0.5
        norm = Normalize(vmin=min_color, vmax=max_color)
        colorbar_ticks = None

    color_mappable: Any
    if n_dimensions == 1:
        scatter = None
        if np.any(constraints_satisfied):
            scatter = ax.scatter(
                np.zeros(np.count_nonzero(constraints_satisfied)),
                rows[constraints_satisfied, 0],
                c=color_values[constraints_satisfied],
                cmap=cmap,
                norm=norm,
                alpha=0.8,
                marker="o",
            )
        if np.any(~constraints_satisfied):
            infeasible_scatter = ax.scatter(
                np.zeros(np.count_nonzero(~constraints_satisfied)),
                rows[~constraints_satisfied, 0],
                c=color_values[~constraints_satisfied],
                cmap=cmap,
                norm=norm,
                alpha=0.8,
                marker="x",
            )
            if scatter is None:
                scatter = infeasible_scatter
        color_mappable = scatter
    else:
        x_values = np.arange(n_dimensions)
        segments = [np.column_stack([x_values, row]) for row in rows]
        feasible_segments = [
            segment for segment, satisfies in zip(segments, constraints_satisfied) if satisfies
        ]
        infeasible_segments = [
            segment for segment, satisfies in zip(segments, constraints_satisfied) if not satisfies
        ]

        feasible_collection = LineCollection(
            feasible_segments,
            cmap=cmap,
            norm=norm,
            alpha=0.5,
            linestyles="solid",
        )
        feasible_collection.set_array(color_values[constraints_satisfied])
        feasible_collection.set_linewidth(1.2)
        ax.add_collection(feasible_collection)
        color_mappable = feasible_collection

        if infeasible_segments:
            infeasible_collection = LineCollection(
                infeasible_segments,
                cmap=cmap,
                norm=norm,
                alpha=0.5,
                linestyles="dotted",
            )
            infeasible_collection.set_array(color_values[~constraints_satisfied])
            infeasible_collection.set_linewidth(1.2)
            ax.add_collection(infeasible_collection)
            if not feasible_segments:
                color_mappable = infeasible_collection

    colorbar = fig.colorbar(color_mappable, pad=0.1, ax=ax, ticks=colorbar_ticks)
    colorbar.set_label(info.color_label)
    _add_dimension_axes(ax, info.dimensions)
    if np.any(~constraints_satisfied):
        ax.set_title("Parallel Coordinate Plot", pad=32)
        ax.legend(
            handles=[
                Line2D([0], [0], color="#2F3B52", linestyle="solid", label="Feasible"),
                Line2D([0], [0], color="#2F3B52", linestyle="dotted", label="Infeasible"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=2,
            frameon=False,
        )

    return ax


def _add_dimension_axes(ax: Any, dimensions: list[Any]) -> None:
    n_dimensions = len(dimensions)
    for index, dimension in enumerate(dimensions):
        ax2 = ax.twinx()
        ax2.set_ylim(0.0, 1.0)
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.xaxis.set_visible(False)
        ax2.yaxis.set_ticks_position("right")
        ax2.spines["right"].set_position(("axes", _axis_fraction(index, n_dimensions)))
        ax2.set_yticks(dimension.tickvals)
        ax2.set_yticklabels(dimension.ticktext)
        ax2.tick_params(axis="y", labelsize=8, length=3, pad=2)


def _axis_fraction(index: int, n_dimensions: int) -> float:
    if n_dimensions == 1:
        return 0.5
    return index / (n_dimensions - 1)
