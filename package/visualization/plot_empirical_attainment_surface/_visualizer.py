from __future__ import annotations

from copy import deepcopy
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-300


def _change_scale(ax: plt.Axes, log_scale_inds: list[int] | None) -> None:
    log_scale_inds = [] if log_scale_inds is None else log_scale_inds
    if 0 in log_scale_inds:
        ax.set_xscale("log")
    if 1 in log_scale_inds:
        ax.set_yscale("log")


def _get_slighly_expanded_value_range(
    loss_vals: np.ndarray, log_scale_inds: list[int] | None = None
) -> tuple[float, float, float, float]:
    X = loss_vals[..., 0].ravel()
    Y = loss_vals[..., 1].ravel()
    log_scale_inds = log_scale_inds if log_scale_inds is not None else []
    x_is_log, y_is_log = 0 in log_scale_inds, 1 in log_scale_inds

    X = X[np.isfinite(X) & (X > EPS) if x_is_log else np.isfinite(X)]
    Y = Y[np.isfinite(Y) & (Y > EPS) if y_is_log else np.isfinite(Y)]
    x_min, x_max, y_min, y_max = X.min(), X.max(), Y.min(), Y.max()
    if x_is_log:
        x_min, x_max = np.log(x_min), np.log(x_max)
    if y_is_log:
        y_min, y_max = np.log(y_min), np.log(y_max)

    x_min -= 0.1 * (x_max - x_min)
    x_max += 0.1 * (x_max - x_min)
    y_min -= 0.1 * (y_max - y_min)
    y_max += 0.1 * (y_max - y_min)
    if x_is_log:
        x_min, x_max = np.exp(x_min), np.exp(x_max)
    if y_is_log:
        y_min, y_max = np.exp(y_min), np.exp(y_max)
    return x_min, x_max, y_min, y_max


def _check_surface(surf: np.ndarray) -> np.ndarray:
    if len(surf.shape) != 2:
        raise ValueError(f"The shape of surf must be (n_points, n_obj), but got {surf.shape}")

    X = surf[:, 0]
    if np.any(np.maximum.accumulate(X) != X):
        raise ValueError("The axis [:, 0] of surf must be an increasing sequence")


def _step_direction(larger_is_better_objectives: list[int] | None) -> str:
    """
    Check here:
        https://matplotlib.org/stable/gallery/lines_bars_and_markers/step_demo.html#sphx-glr-gallery-lines-bars-and-markers-step-demo-py

    min x min (post)
        o...       R
           :
           o...
              :
              o

    max x max (pre)
        o
        :
        ...o
           :
    R      ...o

    min x max (post)
              o
              :
           o...
           :
        o...       R

    max x min (pre)
    R      ...o
           :
        ...o
        :
        o
    """
    if larger_is_better_objectives is None:
        larger_is_better_objectives = []

    large_f1_is_better = bool(0 in larger_is_better_objectives)
    return "pre" if large_f1_is_better else "post"


def _extract_marker_kwargs(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    marker_kwargs: dict[str, Any] = dict()
    new_kwargs: dict[str, Any] = dict()
    for k, v in kwargs.items():
        if k.startswith("mark"):
            marker_kwargs[k] = v
        else:
            new_kwargs[k] = v

    return marker_kwargs, new_kwargs


class EmpiricalAttainmentFuncPlot:
    """
    The class to plot empirical attainment function.

    Args:
        larger_is_better_objectives:
            The indices of the objectives that are better when the values are larger.
            If None, we consider all objectives are better when they are smaller.
        log_scale_inds:
            The indices of the log scale.
            For example, if you would like to plot the first objective in the log scale,
            you need to feed log_scale_inds=[0].
            In principle, log_scale_inds changes the minimum value of the axes
            from -np.inf to a small positive value.
        x_min, x_max, y_min, y_max:
            The lower/upper bounds for each objective if available.
            It is used to fix the value ranges of each objective in plots.
    """

    def __init__(
        self,
        larger_is_better_objectives: list[int] | None = None,
        log_scale_inds: list[int] | None = None,
        x_min: float = np.inf,
        x_max: float = -np.inf,
        y_min: float = np.inf,
        y_max: float = -np.inf,
    ):
        self.step_dir = _step_direction(larger_is_better_objectives)
        self.log_scale_inds = log_scale_inds if log_scale_inds is not None else []
        self.x_is_log, self.y_is_log = 0 in self.log_scale_inds, 1 in self.log_scale_inds
        # We cannot plot until we call _transform_surface_list
        self.x_min = max(EPS, x_min) if 0 in self.log_scale_inds else x_min
        self.x_max = x_max
        self.y_min = max(EPS, y_min) if 1 in self.log_scale_inds else y_min
        self.y_max = y_max

    def _transform_surface_list(self, surfs_list: list[np.ndarray]) -> list[np.ndarray]:
        for surf in surfs_list:
            x_min, x_max, y_min, y_max = _get_slighly_expanded_value_range(
                surf, self.log_scale_inds
            )
            self.x_min, self.x_max = min(self.x_min, x_min), max(self.x_max, x_max)
            self.y_min, self.y_max = min(self.y_min, y_min), max(self.y_max, y_max)

        for surf in surfs_list:
            lb = EPS if self.x_is_log else -np.inf
            surf[..., 0][surf[..., 0] == lb] = self.x_min
            surf[..., 0][surf[..., 0] == np.inf] = self.x_max

            lb = EPS if self.y_is_log else -np.inf
            surf[..., 1][surf[..., 1] == lb] = self.y_min
            surf[..., 1][surf[..., 1] == np.inf] = self.y_max

        return surfs_list

    def plot_surface(
        self,
        ax: plt.Axes,
        surf: np.ndarray,
        color: str | None = None,
        label: str | None = None,
        linestyle: str | None = None,
        marker: str | None = None,
        transform: bool = True,
        **kwargs: Any,
    ) -> matplotlib.lines.Line2D:
        """
        Plot multiple surfaces.

        Args:
            ax:
                The subplots axes.
            surf:
                The vertices of the empirical attainment surface.
                The shape must be (X.size, 2).
            color:
                The color of the plot.
            label:
                The label of the plot.
            linestyle:
                The linestyle of the plot.
            marker:
                The marker of the plot.
            transform:
                Whether automatically transforming based on (x_min, x_max) and (y_min, y_max).
            kwargs:
                The kwargs for ax.plot.

        Returns:
            line:
                The plotted line object.
        """
        if len(surf.shape) != 2 or surf.shape[1] != 2:
            raise ValueError(f"The shape of surf must be (n_points, 2), but got {surf.shape}")

        _surf = surf.copy()
        if transform:
            _surf = self._transform_surface_list(surfs_list=[_surf])[0]
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)

        kwargs.update(drawstyle=f"steps-{self.step_dir}")
        _check_surface(_surf)
        X, Y = _surf[:, 0], _surf[:, 1]
        line = ax.plot(
            X, Y, color=color, label=label, linestyle=linestyle, marker=marker, **kwargs
        )[0]
        _change_scale(ax, self.log_scale_inds)
        return line

    def plot_multiple_surface(
        self,
        ax: plt.Axes,
        surfs: np.ndarray | list[np.ndarray],
        colors: list[str | None] | None = None,
        labels: list[str | None] | None = None,
        linestyles: list[str | None] | None = None,
        markers: list[str | None] | None = None,
        **kwargs: Any,
    ) -> list[matplotlib.lines.Line2D]:
        """
        Plot multiple surfaces.

        Args:
            ax:
                The subplots axes.
            surfs:
                The vertices of the empirical attainment surfaces for each plot.
                Each element should have the shape of (X.size, 2).
                If this is an array, then the shape must be (n_surf, X.size, 2).
            colors:
                The colors of each plot
            labels:
                The labels of each plot.
            linestyles:
                The linestyles of each plot.
            markers:
                The markers of each plot.
            kwargs:
                The kwargs for ax.plot.

        Returns:
            lines:
                A list of the plotted line objects.
        """
        lines: list[matplotlib.lines.Line2D] = []
        _surfs = deepcopy(surfs)
        _surfs = self._transform_surface_list(_surfs)

        n_surfs = len(_surfs)
        linestyles = linestyles if linestyles is not None else [None] * n_surfs
        markers = markers if markers is not None else [None] * n_surfs
        colors = colors if colors is not None else [None] * n_surfs
        labels = labels if labels is not None else [None] * n_surfs
        for surf, color, label, linestyle, marker in zip(
            _surfs, colors, labels, linestyles, markers
        ):
            kwargs.update(color=color, label=label, linestyle=linestyle, marker=marker)
            line = self.plot_surface(ax, surf, transform=False, **kwargs)
            lines.append(line)

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        return lines

    def plot_surface_with_band(
        self,
        ax: plt.Axes,
        surfs: np.ndarray,
        color: str | None = None,
        label: str | None = None,
        linestyle: str | None = None,
        marker: str | None = None,
        transform: bool = True,
        **kwargs: Any,
    ) -> matplotlib.lines.Line2D:
        """
        Plot the surface with a band.
        Typically, we would like to plot median with the band between
        25% -- 75% percentile attainment surfaces.

        Args:
            ax:
                The subplots axes.
            surfs:
                The vertices of the empirical attainment surfaces for each level.
                If surf[i, j, 1] takes np.inf, this is not actually on the surface.
                The shape is (3, X.size, 2).
            color:
                The color of the plot
            label:
                The label of the plot.
            linestyle:
                The linestyle of the plot.
            marker:
                The marker of the plot.
            transform:
                Whether automatically transforming based on (x_min, x_max) and (y_min, y_max).
            kwargs:
                The kwargs for ax.plot.

        Returns:
            line:
                The plotted line object.
        """
        if surfs.shape[0] != 3:
            raise ValueError(
                f"plot_surface_with_band requires three levels, but got only {surfs.shape[0]} levels"
            )

        _surfs = deepcopy(surfs)
        if transform:
            _surfs = self._transform_surface_list(surfs_list=_surfs)
            ax.set_xlim(self.x_min, self.x_max)
            ax.set_ylim(self.y_min, self.y_max)

        for surf in _surfs:
            _check_surface(surf)

        X = _surfs[0, :, 0]

        marker_kwargs, kwargs = _extract_marker_kwargs(**kwargs)
        if color is not None:
            kwargs.update(color=color)
        alpha = kwargs.pop("alpha", None)
        ax.fill_between(
            X, _surfs[0, :, 1], _surfs[2, :, 1], alpha=0.2, step=self.step_dir, **kwargs
        )
        kwargs["alpha"] = alpha

        # marker and linestyle are only for plot
        kwargs.update(label=label, linestyle=linestyle, marker=marker, **marker_kwargs)
        line = ax.plot(X, _surfs[1, :, 1], drawstyle=f"steps-{self.step_dir}", **kwargs)[0]
        _change_scale(ax, self.log_scale_inds)
        return line

    def plot_multiple_surface_with_band(
        self,
        ax: plt.Axes,
        surfs_list: np.ndarray | list[np.ndarray],
        colors: list[str | None] | None = None,
        labels: list[str | None] | None = None,
        linestyles: list[str | None] | None = None,
        markers: list[str | None] | None = None,
        **kwargs: Any,
    ) -> list[matplotlib.lines.Line2D]:
        """
        Plot multiple surfaces with a band.

        Args:
            ax:
                The subplots axes.
            surfs_list:
                The vertices of the empirical attainment surfaces for each plot.
                Each element should have the shape of (3, X.size, 2).
                If this is an array, then the shape must be (n_surf, 3, X.size, 2).
            colors:
                The colors of each plot.
            labels:
                The labels of each plot.
            linestyles:
                The linestyles of each plot.
            markers:
                The markers of each plot.
            kwargs:
                The kwargs for ax.plot.

        Returns:
            lines:
                A list of the plotted line objects.
        """
        lines: list[matplotlib.lines.Line2D] = []
        _surfs_list = deepcopy(surfs_list)
        _surfs_list = self._transform_surface_list(_surfs_list)

        n_surfs = len(_surfs_list)
        linestyles = linestyles if linestyles is not None else [None] * n_surfs
        markers = markers if markers is not None else [None] * n_surfs
        colors = colors if colors is not None else [None] * n_surfs
        labels = labels if labels is not None else [None] * n_surfs
        for surf, color, label, linestyle, marker in zip(
            _surfs_list, colors, labels, linestyles, markers
        ):
            kwargs.update(color=color, label=label, linestyle=linestyle, marker=marker)
            line = self.plot_surface_with_band(ax, surf, transform=False, **kwargs)
            lines.append(line)

        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        return lines
