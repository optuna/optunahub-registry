from __future__ import annotations

from collections.abc import Sequence

import optuna
from optuna.visualization import plot_hypervolume_history as optuna_plot_hypervolume_history
import plotly.graph_objects as go


def plot_hypervolume_history(study: optuna.study, reference_point: Sequence[float]) -> "go.Figure":
    """Plot the study's history as a hypervolume.

    Args:
        study:
            An Optuna study.
        reference_point:
            A reference point of the hypervolume indicator.
    """

    fig = optuna_plot_hypervolume_history(study, reference_point)
    fig.update_layout(
        title=f"Hypervolume History Plot (reference point: {reference_point})",
    )
    return fig
