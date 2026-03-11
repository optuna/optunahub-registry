from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.trial import FrozenTrial


if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure


def _get_param_values_and_objectives(
    study: optuna.Study,
    params: Sequence[str] | None,
    target: Callable[[FrozenTrial], float] | None,
) -> tuple[list[str], dict[str, np.ndarray], np.ndarray]:
    """Extract parameter values and objective values from completed trials."""
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(trials) == 0:
        raise ValueError("The study has no completed trials.")

    if target is None:

        def target(t: FrozenTrial) -> float:
            return t.value  # type: ignore[return-value]

    objectives = np.array([target(t) for t in trials])

    # Determine parameters to plot.
    if params is None:
        all_param_names: set[str] = set()
        for t in trials:
            all_param_names.update(t.params.keys())
        param_names = sorted(all_param_names)
    else:
        param_names = list(params)

    # Collect parameter values (numeric only; categoricals are label-encoded).
    param_values: dict[str, np.ndarray] = {}
    for name in param_names:
        vals: list[float] = []
        mask: list[bool] = []
        for t in trials:
            if name in t.params:
                v = t.params[name]
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                    mask.append(True)
                elif isinstance(v, str):
                    # Label-encode categorical values.
                    vals.append(float(hash(v) % 10000))
                    mask.append(True)
                else:
                    vals.append(0.0)
                    mask.append(False)
            else:
                vals.append(0.0)
                mask.append(False)
        arr = np.array(vals)
        m = np.array(mask)
        if m.sum() > 0:
            param_values[name] = arr

    # Filter param_names to those with valid values.
    param_names = [n for n in param_names if n in param_values]
    return param_names, param_values, objectives


def _compute_density_jitter(
    x: np.ndarray,
    *,
    nbins: int = 50,
    jitter_scale: float = 0.4,
    seed: int = 0,
) -> np.ndarray:
    """Compute density-based jitter for beeswarm layout.

    This is the core of the beeswarm plot: points in dense regions of the
    x-axis get larger y-jitter, creating the characteristic "swarm" shape
    similar to SHAP beeswarm plots.

    The approach uses histogram-based density estimation to avoid a scipy
    dependency, then applies Gaussian jitter scaled by the local density.

    Args:
        x: 1-D array of x-positions (e.g. objective values).
        nbins: Number of bins for density estimation.
        jitter_scale: Maximum jitter magnitude (in row-index units).
        seed: Random seed for reproducibility.

    Returns:
        1-D array of y-offsets.
    """
    n = len(x)
    if n <= 1:
        return np.zeros(n)

    # Histogram-based density estimation.
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-12:
        # All x values are identical; uniform jitter.
        rng = np.random.default_rng(seed)
        return rng.uniform(-jitter_scale * 0.5, jitter_scale * 0.5, n)

    hist, bin_edges = np.histogram(x, bins=nbins)
    bin_indices = np.digitize(x, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, len(hist) - 1)
    density = hist[bin_indices].astype(float)

    # Normalize density to [0, 1].
    d_max = density.max()
    if d_max > 0:
        density /= d_max

    # Apply Gaussian jitter scaled by density.
    rng = np.random.default_rng(seed)
    raw_jitter = rng.standard_normal(n)
    # Clip to avoid extreme outliers.
    raw_jitter = np.clip(raw_jitter, -2.5, 2.5) / 2.5
    return raw_jitter * density * jitter_scale


def _sort_params_by_importance(
    study: optuna.Study,
    param_names: list[str],
    param_values: dict[str, np.ndarray],
    objectives: np.ndarray,
) -> list[str]:
    """Sort parameters by importance (most important at top of plot).

    Tries Optuna's built-in fANOVA-based importance first (handles non-monotonic
    relationships). Falls back to Spearman rank correlation if sklearn is not
    installed.
    """
    # Try Optuna's fANOVA (requires sklearn).
    try:
        importances = optuna.importance.get_param_importances(study)
        return sorted(param_names, key=lambda n: importances.get(n, 0.0))
    except (ImportError, RuntimeError):
        pass

    # Fallback: absolute Spearman rank correlation.
    correlations: dict[str, float] = {}
    for name in param_names:
        vals = param_values[name]
        valid = np.isfinite(vals) & np.isfinite(objectives)
        if valid.sum() < 3:
            correlations[name] = 0.0
            continue
        v = vals[valid]
        o = objectives[valid]
        v_rank = np.argsort(np.argsort(v)).astype(float)
        o_rank = np.argsort(np.argsort(o)).astype(float)
        v_rank -= v_rank.mean()
        o_rank -= o_rank.mean()
        denom = np.sqrt((v_rank**2).sum() * (o_rank**2).sum())
        if denom < 1e-12:
            correlations[name] = 0.0
        else:
            correlations[name] = abs(float(np.dot(v_rank, o_rank) / denom))

    return sorted(param_names, key=lambda n: correlations.get(n, 0.0))


def plot_beeswarm(
    study: optuna.Study,
    *,
    params: Sequence[str] | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
    color_map: str = "RdBu_r",
    ax: Axes | None = None,
) -> tuple[Figure, Axes, Colorbar]:
    """Plot a SHAP-style beeswarm plot for an Optuna study.

    Each row represents a hyperparameter. Each dot is one trial, positioned
    on the x-axis by its objective value. The dot color represents the
    parameter value (blue = low, red = high). In dense regions, dots are
    spread vertically to reveal the distribution (beeswarm layout).

    This visualization is useful for understanding monotonic relationships
    between hyperparameter values and the objective function.

    Args:
        study:
            An Optuna study with completed trials.
        params:
            A list of parameter names to include. If ``None``, all parameters
            across completed trials are used.
        target:
            A callable that extracts a scalar value from a
            :class:`~optuna.trial.FrozenTrial`. Defaults to
            ``trial.value``.
        target_name:
            Label for the x-axis. Defaults to ``"Objective Value"``.
        color_map:
            Matplotlib colormap name for parameter value coloring.
            Defaults to ``"RdBu_r"`` (blue for low, red for high values).
        ax:
            Matplotlib axes to draw on. If ``None``, a new figure is created.

    Returns:
        A tuple of ``(figure, axes, colorbar)``.

    Raises:
        ValueError: If the study has no completed trials.
    """
    param_names, param_values, objectives = _get_param_values_and_objectives(study, params, target)
    if len(param_names) == 0:
        raise ValueError("No valid parameters found in completed trials.")

    # Sort parameters by importance (least important at bottom).
    sorted_params = _sort_params_by_importance(study, param_names, param_values, objectives)

    # Resolve colormap.
    cmap: Colormap = cm.get_cmap(color_map)

    # Create figure if needed.
    if ax is None:
        n_params = len(sorted_params)
        fig_height = max(3.0, 0.5 * n_params + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_height))
    else:
        fig = ax.get_figure()

    # Plot each parameter as a row.
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    for row_idx, param_name in enumerate(sorted_params):
        vals = param_values[param_name]

        # Normalize parameter values to [0, 1] for coloring.
        v_min, v_max = vals.min(), vals.max()
        if v_max - v_min < 1e-12:
            norm_vals = np.full_like(vals, 0.5)
        else:
            norm_vals = (vals - v_min) / (v_max - v_min)

        # Compute density-based jitter.
        jitter = _compute_density_jitter(objectives, seed=row_idx)
        y_positions = row_idx + jitter

        # Map normalized values to colors.
        colors = cmap(norm_vals)

        ax.scatter(
            objectives,
            y_positions,
            c=colors,
            s=8,
            alpha=0.75,
            edgecolors="none",
            rasterized=True,
        )

    # Configure axes.
    ax.set_yticks(range(len(sorted_params)))
    ax.set_yticklabels(sorted_params, fontsize=14)
    ax.set_xlabel(target_name, fontsize=14)
    ax.set_title("Beeswarm Plot", fontsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=12)

    # Add colorbar.
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Parameter value (normalized)", fontsize=12)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Low", "Mid", "High"])
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    return fig, ax, cbar


__all__ = ["plot_beeswarm"]
