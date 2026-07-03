from __future__ import annotations

from collections.abc import Callable
from typing import Optional
from typing import TYPE_CHECKING

from optuna.study import Study
from optuna.trial import FrozenTrial

from ._data import _get_plot_info
from ._data import _PlotInfo
from ._data import DEFAULT_MISSING_LABEL


if TYPE_CHECKING:
    import plotly.graph_objects as go


def plot_parallel_coordinate(
    study: Study,
    params: Optional[list[str]] = None,
    *,
    target: Optional[Callable[[FrozenTrial], float]] = None,
    target_name: str = "Objective Value",
    objective_names: Optional[list[str]] = None,
    missing_label: str = DEFAULT_MISSING_LABEL,
) -> "go.Figure":
    """Plot a parallel coordinate plot with Plotly.

    This plot differs from Optuna's built-in parallel coordinate plot in two ways:

    * Conditional parameters are supported. If a trial does not contain a selected parameter,
      the trial is connected to a special ``missing_label`` tick below the valid values.
    * Multi-objective studies are supported without a ``target`` callback. Objective axes are
      placed side by side, and lines are colored by Pareto rank.

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
        A :class:`plotly.graph_objects.Figure` object.
    """

    info = _get_plot_info(study, params, target, target_name, objective_names, missing_label)
    return _get_parallel_coordinate_plot(info)


def _get_parallel_coordinate_plot(info: _PlotInfo) -> "go.Figure":
    import plotly.graph_objects as go

    layout = go.Layout(title="Parallel Coordinate Plot")
    if len(info.dimensions) == 0 or len(info.color_values) == 0:
        return go.Figure(data=[], layout=layout)

    if not all(info.constraints_satisfied):
        return _get_parallel_coordinate_scatter_plot(info)

    dimensions = []
    for dimension in info.dimensions:
        values = [dimension.values[trial_id] for trial_id in info.draw_order]
        dimensions.append(
            {
                "label": dimension.label,
                "range": [0.0, 1.0],
                "values": values,
                "tickvals": list(dimension.tickvals),
                "ticktext": list(dimension.ticktext),
            }
        )

    color_values = [info.color_values[trial_id] for trial_id in info.draw_order]
    line = {
        "color": color_values,
        "colorscale": "Blues",
        "colorbar": {"title": info.color_label},
        "showscale": True,
        "reversescale": info.reverse_scale,
    }
    if info.is_rank_color:
        max_rank = max(color_values)
        line.update(
            {
                "cmin": -0.5,
                "cmax": max_rank + 0.5,
                "colorbar": {
                    "title": info.color_label,
                    "tickvals": list(range(int(max_rank) + 1)),
                },
            }
        )

    return go.Figure(
        data=[
            go.Parcoords(
                dimensions=dimensions,
                labelangle=30,
                labelside="bottom",
                line=line,
            )
        ],
        layout=layout,
    )


def _get_parallel_coordinate_scatter_plot(info: _PlotInfo) -> "go.Figure":
    from plotly.colors import sample_colorscale
    import plotly.graph_objects as go

    n_dimensions = len(info.dimensions)
    x_values = list(range(n_dimensions))
    cmin, cmax = _get_color_range(info)
    data = []

    for trial_id in info.draw_order:
        color_value = info.color_values[trial_id]
        line_color = sample_colorscale(
            "Blues", [_normalize_color_value(color_value, cmin, cmax, info.reverse_scale)]
        )[0]
        data.append(
            go.Scatter(
                x=x_values,
                y=[dimension.values[trial_id] for dimension in info.dimensions],
                mode="lines",
                line={
                    "color": line_color,
                    "width": 1.2,
                    "dash": "solid" if info.constraints_satisfied[trial_id] else "dot",
                },
                opacity=0.5,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    colorbar: dict[str, object] = {"title": info.color_label}
    colorbar["x"] = 1.04
    if info.is_rank_color:
        colorbar["tickvals"] = list(range(int(max(info.color_values)) + 1))
    data.append(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "color": [cmin, cmax],
                "cmin": cmin,
                "cmax": cmax,
                "colorscale": "Blues",
                "reversescale": info.reverse_scale,
                "showscale": True,
                "colorbar": colorbar,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )
    data.extend(_get_constraint_legend_traces())

    return go.Figure(
        data=data,
        layout=go.Layout(
            xaxis={
                "range": [-0.1, max(n_dimensions - 0.9, 0.5)],
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
            },
            yaxis={
                "domain": [0.0, 0.94],
                "range": [-0.12, 1.05],
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
            },
            shapes=_get_axis_shapes(n_dimensions),
            annotations=[_get_title_annotation()] + _get_axis_annotations(info),
            showlegend=True,
            legend={
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": 1.07,
                "yanchor": "top",
            },
            margin={"t": 72},
            plot_bgcolor="white",
            paper_bgcolor="white",
        ),
    )


def _get_title_annotation() -> dict[str, object]:
    return {
        "xref": "paper",
        "yref": "paper",
        "x": 0.5,
        "y": 1.13,
        "text": "Parallel Coordinate Plot",
        "showarrow": False,
        "xanchor": "center",
        "yanchor": "top",
        "font": {"size": 17},
    }


def _get_color_range(info: _PlotInfo) -> tuple[float, float]:
    if info.is_rank_color:
        return -0.5, max(info.color_values) + 0.5

    cmin = min(info.color_values)
    cmax = max(info.color_values)
    if cmin == cmax:
        return cmin - 0.5, cmax + 0.5
    return cmin, cmax


def _normalize_color_value(value: float, cmin: float, cmax: float, reverse_scale: bool) -> float:
    normalized = 0.5 if cmin == cmax else (value - cmin) / (cmax - cmin)
    normalized = min(max(normalized, 0.0), 1.0)
    return 1.0 - normalized if reverse_scale else normalized


def _get_axis_shapes(n_dimensions: int) -> list[dict[str, object]]:
    return [
        {
            "type": "line",
            "xref": "x",
            "yref": "y",
            "x0": index,
            "x1": index,
            "y0": 0.0,
            "y1": 1.0,
            "line": {"color": "#B8B8B8", "width": 1},
        }
        for index in range(n_dimensions)
    ]


def _get_axis_annotations(info: _PlotInfo) -> list[dict[str, object]]:
    annotations = []
    n_dimensions = len(info.dimensions)
    for index, dimension in enumerate(info.dimensions):
        for tick, text in zip(dimension.tickvals, dimension.ticktext):
            annotations.append(
                {
                    "xref": "x",
                    "yref": "y",
                    "x": index,
                    "y": tick,
                    "text": text,
                    "showarrow": False,
                    "xanchor": "right" if index == 0 else "left",
                    "yanchor": "middle",
                    "xshift": -6 if index == 0 else 6,
                    "font": {"size": 11},
                }
            )
        annotations.append(
            {
                "xref": "x",
                "yref": "y",
                "x": index,
                "y": -0.08,
                "text": dimension.label,
                "showarrow": False,
                "textangle": 30,
                "xanchor": "right" if index < n_dimensions - 1 else "center",
                "yanchor": "top",
                "font": {"size": 12},
            }
        )
    return annotations


def _get_constraint_legend_traces() -> list["go.Scatter"]:
    import plotly.graph_objects as go

    return [
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": "#2F3B52", "dash": "solid", "width": 2},
            name="Feasible",
            hoverinfo="skip",
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line={"color": "#2F3B52", "dash": "dot", "width": 2},
            name="Infeasible",
            hoverinfo="skip",
        ),
    ]
