from __future__ import annotations

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy.stats import linregress


if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    return x**2


def _extract_elapsed_times(studies: dict[str, list[optuna.Study]]) -> dict[str, list[list[float]]]:
    elapsed_times: dict[str, list[list[float]]] = {}
    for label, target_studies in studies.items():
        elapsed_times[label] = []
        for study in target_studies:
            trials = study.get_trials(
                states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
            )
            start_time = trials[0].datetime_start
            assert start_time is not None
            ets = [
                (t.datetime_complete - start_time).total_seconds()
                for t in trials
                if t.datetime_complete is not None
            ]
            elapsed_times[label].append(ets)
    return elapsed_times


def _extract_n_completed_trials(
    elapsed_times: dict[str, list[list[float]]], time_grids: np.ndarray
) -> dict[str, np.ndarray]:
    n_completed_trials: dict[str, list[np.ndarray]] = {}
    for label, target_elapsed_times in elapsed_times.items():
        n_completed_trials[label] = []
        for tried_elapsed_times in target_elapsed_times:
            n_completed_trials[label].append(
                np.searchsorted(tried_elapsed_times, time_grids, side="left")
            )

    return {label: np.mean(v, axis=0) for label, v in n_completed_trials.items()}


def _predict_n_completed_trials(
    time_grids: np.ndarray, n_completed_trials_stats: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    for label, mean in n_completed_trials_stats.items():
        use_train = np.ones_like(mean, dtype=bool)
        use_train[1:] = mean[:-1] != mean[1:]
        slope, intercept, _, _, _ = linregress(
            np.log(time_grids[use_train]), np.log(mean[use_train])
        )
        preds[label] = np.exp(slope * np.log(time_grids) + intercept)

    return preds


def plot_sampling_speed(studies: dict[str, list[optuna.Study]], ax: Axes | None = None) -> Axes:
    elapsed_times = _extract_elapsed_times(studies)
    max_time = max(
        max(max(t) for t in target_elapsed_times)
        for target_elapsed_times in elapsed_times.values()
    )
    max_time_exponent_of_10 = int(math.ceil(math.log(max_time) / math.log(10)))
    step_size = (max_time_exponent_of_10 + 2) * 10 + 1
    time_grids = 10 ** np.linspace(-2, max_time_exponent_of_10, step_size)
    last_grid_index = np.searchsorted(time_grids, max_time)
    time_grids = time_grids[: last_grid_index + 1]
    n_completed_trials_stats = _extract_n_completed_trials(elapsed_times, time_grids)
    if ax is None:
        _, ax = plt.subplots()

    preds = _predict_n_completed_trials(time_grids, n_completed_trials_stats)
    for label, n_completed_trials in n_completed_trials_stats.items():
        use_for_plot = np.ones_like(n_completed_trials, dtype=bool)
        use_for_plot[1:] = n_completed_trials[:-1] != n_completed_trials[1:]
        ax.scatter(
            n_completed_trials[use_for_plot][1:-1],
            time_grids[use_for_plot][1:-1],
            label=label,
            edgecolor="black",
        )
        ax.plot(preds[label], time_grids, linestyle="dotted")

    ax.set_title("Elapsed Time at Each Trial")
    ax.grid(which="minor", color="gray", linestyle=":")
    ax.grid(which="major", color="black")
    ax.set_xlabel("Number of Sampled Trials")
    ax.set_xscale("log")
    ax.set_ylim(
        time_grids[0] - (time_grids[1] - time_grids[0]) / 2, (time_grids[-2] + time_grids[-1]) / 2
    )
    ax.set_ylabel("Elapsed Time [s]")
    ax.set_yscale("log")
    ax.legend()
    return ax


__all__ = ["plot_sampling_speed"]
