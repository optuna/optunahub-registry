from __future__ import annotations

from optuna.study import Study
from optuna.trial import TrialState
import plotly.graph_objects as go


def plot_step_distribution(study: Study) -> go.Figure:
    """Plot the distribution of the steps in the study.

    Args:
        study: The study to plot.

    Returns:
        The plotly figure.
    """
    trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED))
    last_steps = [trial.last_step for trial in trials]
    max_step = max([s for s in last_steps if s is not None])

    scattered = go.Scatter(
        x=[trial.number for trial in trials],
        y=[s if s is not None else max_step for s in last_steps],
        mode="markers",
        marker=dict(size=3, color=[trial.value for trial in trials], colorscale="Viridis"),
    )
    layout = go.Layout(
        title="Step distribution",
        xaxis=dict(title="Trial number"),
        yaxis=dict(title="Step"),
        showlegend=False,
    )
    return go.Figure(data=[scattered], layout=layout)
