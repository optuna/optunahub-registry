"""
CMA-ES visualization adapted from CyberAgentAILab/cmaes.

Please Refer for More details:
    Masahiro Nomura, Masashi Shibata, Ryoki Hamano.
    cmaes: A Simple yet Practical Python Library for CMA-ES
    https://arxiv.org/abs/2402.01373
    git hub: https://github.com/CyberAgentAILab/cmaes

"""

from __future__ import annotations

from collections.abc import Callable
import pickle
from typing import Any
from typing import Sequence
from typing import TYPE_CHECKING

from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from optuna.distributions import FloatDistribution
from optuna.samplers import CmaEsSampler
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from pylab import rcParams
from scipy import stats


if TYPE_CHECKING:
    from optuna.study import Study


color_dict = {
    "red": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "green": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "blue": ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
    "yellow": ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
}

bw = LinearSegmentedColormap("Bluewhile", color_dict)

rcParams["figure.figsize"] = 10, 5


unsupported: tuple[str, ...] = ("sepcma", "cmawm", "restart", "cma:ipop", "cma:bipop")


def get_trial_values(
    study: Study,
    params: Sequence[str] | None,
    target: Callable[[FrozenTrial], float] | None,
) -> tuple[np.ndarray, list[tuple[float, float]], list[str], np.ndarray]:
    """Extract objective values, bounds, parameter name and parameter values from the completed trials."""

    sampler = study.sampler

    if not isinstance(sampler, CmaEsSampler):
        raise TypeError("Plot CMAE is only comptabile with sampler.CmaEsSampler")

    if study._is_multi_objective():
        raise ValueError("Plot CMAE doesn't support multi-objectives studies")

    states = (TrialState.COMPLETE,)
    trials = study._storage.get_all_trials(study._study_id, deepcopy=False, states=states)

    if len(trials) < 1:
        raise ValueError("The study has no completed trials.")

    if any(k.startswith(unsupported) for t in trials for k in t.system_attrs):
        raise ValueError(
            "Plot Cmae only supports default Cma Es. 'use_Seperable_cma = True' or "
            "'with_margin = True' not supported yet"
        )

    if not any(("cma:generation") in t.system_attrs for t in trials):
        raise ValueError("No CMA Objects exist in the study")

    if target is None:

        def target(t: FrozenTrial) -> float:
            return t.value

    objectives = np.array([target(t) for t in trials])

    param_dist: dict[str, Any] = {}

    for t in trials:
        for param_name, dist in t.distributions.items():
            param_dist.setdefault(param_name, dist)

    if params is None:
        params = sorted(set(param_dist.keys()))

    if len(params) != 2:
        raise ValueError(
            "Plot CMAE only supports 2 parameters. Please specify exactly 2 parameters to plot."
        )

    # validate parameter names
    for name in params:
        if name not in param_dist:
            raise ValueError(
                f"Parameter '{name}' not found in study."
                f" Available parameters are : {list(param_dist.keys())}"
            )

    bounds: list[tuple[float, float]] = []

    for param_name in params:
        dist = param_dist[param_name]

        if not isinstance(dist, FloatDistribution):
            raise TypeError(
                f"Parameter {param_name} has a distribution which is not supported yet. Only FloatDistribution is supported."
            )

        if dist.log or dist.step is not None:
            raise ValueError(
                f"Parameter {param_name} has a log scale or step size, which is not supported yet."
            )

        bounds.append((dist.low, dist.high))

    points = np.array([[t.params[param_name] for param_name in params] for t in trials])

    return objectives, bounds, list(params), points


def concatenate_optimizer_attrs(optimizer_attrs: dict[str, str]) -> str:
    """ "Concatenate the chunk optimizers attributes in numeric order."""

    return "".join(optimizer_attrs[f"cma:optimizer:{i}"] for i in range(len(optimizer_attrs)))


def restore_optimizer_attrs(trial: FrozenTrial) -> Any | None:
    """Restore a CMA-ES optimizer from a single trial's system attributes."""

    optimizer_attrs = {
        key: val for key, val in trial.system_attrs.items() if key.startswith("cma:optimizer:")
    }

    if not optimizer_attrs:
        return None

    optimizer_str = concatenate_optimizer_attrs(optimizer_attrs)
    return pickle.loads(bytes.fromhex(optimizer_str))


def get_cmaes_objects(
    completed_trials: list[FrozenTrial],
) -> list[tuple[np.ndarray, np.ndarray, float, int, int]]:
    """Return mean, covar, sigma, generation and trial_number for every stored optimizer."""

    cma_object = []

    for trial in completed_trials:
        optimizer = restore_optimizer_attrs(trial)

        if optimizer is None:
            continue

        mean = optimizer._mean.copy()
        covar = optimizer._C.copy()
        sigma = optimizer._sigma
        generation = optimizer.generation
        trial_number = trial.number

        cma_object.append((mean, covar, sigma, generation, trial_number))

    if not cma_object:
        raise ValueError("No CMA-ES optimizer found in the completed trials.")

    return cma_object


def transform_gaussian(
    mean: np.ndarray,
    covar: np.ndarray,
    lower: Sequence[float],
    upper: Sequence[float],
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """ "Transform mean and covar from [0,1] space back to the orginial space."""

    upper_array = np.asarray(upper, dtype=float)
    lower_array = np.asarray(lower, dtype=float)

    scale = upper_array - lower_array

    mean_tranformed = lower_array + scale * np.asarray(mean, dtype=float)

    a = np.diag(scale)

    covar_modified = a @ np.asarray(covar, dtype=float) @ a.T

    covar_tranformed = (sigma**2) * covar_modified

    return mean_tranformed, covar_tranformed


def plot_cmaes(
    study: Study,
    params: Sequence[str] | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
    contour_function: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    interval: int = 500,
) -> animation.FuncAnimation:
    """Animate how CMA-ES moves its sampling distribution, one frame per generation.

    Left panel:
         The sampled trials (colored by objective value, or red points on the contour of
        ``contour_function`` if it is given) and the best trial of the study.


    Right panel:
        The density of the sampling distribution N(mean, sigma^2 C).

    note:
        This function only supports 2-dimensional continuous search spaces.
        It requires the study to use :class:`~optuna.samplers.CmaEsSampler`
        without ``use_seperable_cma = True `` or ``with_margin = True`` or
        ``restart_strategies``.

    Args:
        study :
            A single-objective study optimize with default
            :class:`~optuna.samplers.CmaEsSampler`, ``use_separable_cma = True``,
            ``with_margin = True``,and `` restart_startegy`` are not supported.

        params:
            The names of the two parameters to plot.
            Only float parameters without ``log`` or ``step`` are supported for now.
            If ``None``, all paramters across the completed trials are used.

        target:
            A callable that extracts scalar values from
            :class:`~optuna.trial.FrozenTrial`. Defaults to
            ``trial.value`.

        contour_function:
            Optional ``f(x1, x2)`` drawn as contour lines,
            for experiments where the objective is known.


        interval:
            Delay between frames in milliseconds . Default value 500.


    Return:
         A :class:`matplotlib.animation.FuncAnimation`.

    """

    fig, (ax1, ax2) = plt.subplots(1, 2)

    objective, bounds, params, points = get_trial_values(study, params, target)

    if len(params) != 2:
        raise ValueError(
            "Plot CMAE only supports 2 parameters. Please specify exactly 2 parameters to plot."
        )

    states = (TrialState.COMPLETE,)
    trials = study._storage.get_all_trials(study._study_id, deepcopy=False, states=states)

    if len(trials) < 1:
        raise ValueError("The study has no completed trials.")

    Cma_snaphots = get_cmaes_objects(trials)

    x1_lower_bound, x1_upper_bound = bounds[0]
    x2_lower_bound, x2_upper_bound = bounds[1]

    def init() -> None:

        ax1.clear()
        ax2.clear()

        x1, x2 = params[0], params[1]

        ax1.set_xlim(x1_lower_bound, x1_upper_bound)
        ax1.set_ylim(x2_lower_bound, x2_upper_bound)
        ax2.set_xlim(x1_lower_bound, x1_upper_bound)
        ax2.set_ylim(x2_lower_bound, x2_upper_bound)

        # Plot best params as local minima for overall trial
        best_params = study.best_params
        ax1.plot(best_params[x1], best_params[x2], "y*", ms=10)
        ax2.plot(best_params[x1], best_params[x2], "y*", ms=10)

        # Plot contour of objective function if given
        if contour_function is not None:
            x1 = np.arange(x1_lower_bound, x1_upper_bound, 0.01)
            x2 = np.arange(x2_lower_bound, x2_upper_bound, 0.01)
            x1, x2 = np.meshgrid(x1, x2)
            z = contour_function(x1, x2)  # need to define contour function
            ax1.contour(x1, x2, z, 30, cmap=bw)

    def update(frames: int) -> None:

        current_frame = Cma_snaphots[frames]
        mean, covar, sigma, generation, trial_number = current_frame

        # Boolean masking for seen trials
        seen = np.array([t.number <= trial_number for t in trials])
        points_so_far = points[seen]
        objective_so_far = objective[seen]

        # Plot sample points
        if len(points_so_far) > 0:
            # With Contour Function Known
            if contour_function is not None:
                ax1.plot(
                    points_so_far[:, 0], points_so_far[:, 1], "o", c="r", label="2d", alpha=0.5
                )

            # With Contour Function Not Known
            else:
                scatter = ax1.scatter(
                    points_so_far[:, 0],
                    points_so_far[:, 1],
                    c=objective_so_far,
                    cmap="viridis",
                    label="2d",
                    alpha=0.5,
                    vmin=np.min(objective),
                    vmax=np.max(objective),
                )

                if not fig.axes or len(fig.axes) == 2:
                    fig.colorbar(scatter, ax=ax1, label="Objective value ")

        # Plot multivariate gaussian distribution of CMA-ES
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        mean_tranform, covar_transform = transform_gaussian(mean, covar, lower, upper, sigma)

        x, y = np.mgrid[x1_lower_bound:x1_upper_bound:0.01, x2_lower_bound:x2_upper_bound:0.01]
        pos = np.dstack((x, y))
        rv = stats.multivariate_normal(mean_tranform, covar_transform)

        ax2.contourf(x, y, rv.pdf(pos))

        fig.suptitle(f"CMAES-Optimization Generation {generation} trial = {trial_number}")

    return animation.FuncAnimation(
        fig, update, frames=len(Cma_snaphots), init_func=init, blit=False, interval=interval
    )
