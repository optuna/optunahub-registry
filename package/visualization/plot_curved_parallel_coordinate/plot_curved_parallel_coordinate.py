from __future__ import annotations

from typing import List
from typing import Optional

import numpy as np
import optuna
from optuna.trial import TrialState
from plotly.colors import sample_colorscale
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator


def plot_curved_parallel_coordinate(
    study: optuna.Study, params: Optional[List[str]] = None, points_per_segment: int = 50
) -> go.Figure:
    """Plot a curved parallel coordinate plot for the study."""

    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not trials:
        return go.Figure()

    if params is None:
        params = sorted(list(set(k for t in trials for k in t.params.keys())))

    if len(params) < 1:
        raise ValueError("Curved parallel coordinates require at least one parameter.")

    objectives = [t.value for t in trials]
    min_obj, max_obj = min(objectives), max(objectives)

    plot_dims = ["Objective Value"] + params
    params_bounds = {"Objective Value": (min_obj, max_obj)}

    for p in params:
        vals = [t.params.get(p, np.nan) for t in trials]
        numeric_vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
        if numeric_vals:
            params_bounds[p] = (min(numeric_vals), max(numeric_vals))
        else:
            params_bounds[p] = (0, 1)

    data_matrix = []
    for t in trials:
        row = []
        val = t.value
        norm_obj = (val - min_obj) / (max_obj - min_obj) if max_obj > min_obj else 0.5
        row.append(norm_obj)

        for p in params:
            val = t.params.get(p, np.nan)
            if isinstance(val, (int, float)) and not np.isnan(val):
                p_min, p_max = params_bounds[p]
                norm_val = (val - p_min) / (p_max - p_min) if p_max > p_min else 0.5
                row.append(norm_val)
            else:
                row.append(np.nan)
        data_matrix.append(row)

    fig = go.Figure()
    x_coords = np.arange(len(plot_dims))

    colorscale = "Blues_r"
    for idx, row in enumerate(data_matrix):
        if np.isnan(row).any():
            continue

        interpolator = PchipInterpolator(x_coords, row)
        x_smooth = np.linspace(x_coords.min(), x_coords.max(), len(plot_dims) * points_per_segment)
        y_smooth = interpolator(x_smooth)

        val = objectives[idx]
        norm_obj = (val - min_obj) / (max_obj - min_obj) if max_obj > min_obj else 0.5
        line_color = sample_colorscale(colorscale, [norm_obj])[0]

        fig.add_trace(
            go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode="lines",
                line=dict(color=line_color, width=1.5),
                opacity=0.4,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for i, p in enumerate(plot_dims):
        p_min, p_max = params_bounds[p]
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=1, line=dict(color="#777777", width=1.5))
        fig.add_annotation(x=i, y=1.05, text=p, showarrow=False, font=dict(size=14))
        fig.add_annotation(x=i, y=-0.05, text=f"{p_min:.2f}", showarrow=False)
        fig.add_annotation(x=i, y=1.0, text=f"{p_max:.2f}", showarrow=False, xanchor="left")

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=colorscale,
                cmin=min_obj,
                cmax=max_obj,
                color=[min_obj, max_obj],
                colorbar=dict(title="Objective Value", thickness=20),
                showscale=True,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Curved Parallel Coordinates",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    return fig
