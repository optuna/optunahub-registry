from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import optuna
from ribs.visualize import grid_archive_heatmap


if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes


def plot_grid_archive_heatmap(  # type: ignore
    study: optuna.Study,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Wrapper around pyribs grid_archive_heatmap.

    Args:
        study: Optuna study with a sampler that uses pyribs. This function will
            plot the result archive from the sampler's scheduler.
        ax: Axes on which to plot the heatmap. If None, we create a new axes.
        kwargs: All remaining kwargs will be passed to `grid_archive_heatmap
            <https://docs.pyribs.org/en/stable/api/ribs.visualize.grid_archive_heatmap.html>`_.
    Returns:
        The axes on which the plot was created.
    """
    if ax is None:
        ax = plt.gca()

    archive = study.sampler.scheduler.result_archive
    grid_archive_heatmap(archive, ax=ax, **kwargs)

    return ax


__all__ = ["plot_grid_archive_heatmap"]
