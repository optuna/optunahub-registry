from __future__ import annotations

from collections.abc import Iterator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pytest

from .. import plot_empirical_attainment_surface
from .._eaf import _get_empirical_attainment_surfaces


matplotlib.use("Agg")
optuna.logging.set_verbosity(optuna.logging.ERROR)
DIRECTIONS = [
    ["minimize", "minimize"],
    ["maximize", "minimize"],
    ["minimize", "maximize"],
    ["maximize", "maximize"],
]


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def _create_studies(directions: list[str], n_runs: int, n_trials: int = 30) -> list[optuna.Study]:
    studies = []
    for seed in range(n_runs):
        rng = np.random.RandomState(seed)
        study = optuna.create_study(directions=directions)
        for _ in range(n_trials):
            study.add_trial(
                optuna.trial.create_trial(
                    values=list(rng.rand(2) + 0.01), state=optuna.trial.TrialState.COMPLETE
                )
            )
        studies.append(study)
    return studies


@pytest.mark.parametrize("directions", DIRECTIONS)
def test_surfaces_match_brute_force(directions: list[str]) -> None:
    n_runs = 7
    levels = [1, 4, n_runs]
    studies = _create_studies(directions, n_runs)
    sign = np.array([1.0 if d == "minimize" else -1.0 for d in directions])
    loss_per_run = [sign * np.array([t.values for t in s.trials]) for s in studies]
    surfs = _get_empirical_attainment_surfaces(studies, levels)

    rng = np.random.RandomState(0)
    for level, surf in zip(levels, surfs):
        vertices = sign * surf[np.isfinite(surf).all(axis=1)]
        for p in sign * (rng.rand(500, 2) + 0.01):
            # A point is attained by a run if any of the run's trials weakly dominates it.
            n_attained = sum(np.any(np.all(loss <= p, axis=1)) for loss in loss_per_run)
            on_or_behind_surface = np.any(np.all(vertices <= p, axis=1))
            assert (n_attained >= level) == on_or_behind_surface


@pytest.mark.parametrize("directions", DIRECTIONS)
@pytest.mark.parametrize("log_scale_inds", [None, [0], [1], [0, 1]])
def test_plot_follows_directions_and_scales(
    directions: list[str], log_scale_inds: list[int] | None
) -> None:
    studies = _create_studies(directions, n_runs=5)
    ax = plot_empirical_attainment_surface(studies, [0.5], log_scale_inds=log_scale_inds)
    log_scale_inds = log_scale_inds or []
    assert ax.lines[0].get_drawstyle() == (
        "steps-pre" if directions[0] == "maximize" else "steps-post"
    )
    assert ax.get_xscale() == ("log" if 0 in log_scale_inds else "linear")
    assert ax.get_yscale() == ("log" if 1 in log_scale_inds else "linear")


@pytest.mark.parametrize("directions", DIRECTIONS)
@pytest.mark.parametrize("log_scale_inds", [None, [0], [1], [0, 1]])
def test_plotted_surface_is_finite_and_in_log_range(
    directions: list[str], log_scale_inds: list[int] | None
) -> None:
    studies = _create_studies(directions, n_runs=5)
    ax = plot_empirical_attainment_surface(
        studies, [0.25, 0.5, 0.75], log_scale_inds=log_scale_inds
    )
    log_scale_inds = log_scale_inds or []
    for line in ax.lines:
        for i, vals in enumerate([line.get_xdata(), line.get_ydata()]):
            assert np.all(np.isfinite(vals))
            assert i not in log_scale_inds or np.all(vals > 0)
