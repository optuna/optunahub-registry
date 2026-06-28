from __future__ import annotations

import contextlib
from copy import deepcopy
import logging
from typing import TYPE_CHECKING

import numpy as np
import optuna
from optuna.pruners import BasePruner
from optuna.study import StudyDirection
from optuna.trial import create_trial
from optuna.trial import TrialState

from ._nondomination import _fast_non_domination_rank


if TYPE_CHECKING:
    from collections.abc import Generator

    from optuna.study import Study
    from optuna.trial import FrozenTrial


_USER_ATTR_KEY = "multi_pruner:values"
# Example of the attribute data (ivs; intermediate values):
# {
#     "0": {"loss": 0.52, "acc": 0.81},  # The reported values at step 0.
#     "5": {"loss": 0.31},        # mixed-frequency: only one metric at this step
# }


def _nondomination_rank(
    values_list: list[list[float]], directions: list[StudyDirection]
) -> np.ndarray:
    lvals = np.array(values_list, dtype=float)
    lvals *= np.array(
        [-1.0 if d == StudyDirection.MAXIMIZE else 1.0 for d in directions],
        dtype=float,
    )
    return _fast_non_domination_rank(lvals)


@contextlib.contextmanager
def _suppress_new_study_log() -> Generator[None, None, None]:
    logger = logging.getLogger("optuna.storages._in_memory")
    original_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(original_level)


def _build_synthetic_study_and_trial(
    trial: FrozenTrial,
    trials: list[FrozenTrial],
    ivs: list[dict[int, float]],
    current_ivs: dict[int, float],
    *,
    direction: str | StudyDirection,
) -> tuple[Study, FrozenTrial]:
    with _suppress_new_study_log():
        new_study = optuna.create_study(direction=direction)
    for t, ivs in zip(trials, ivs):
        if not t.state.is_finished() or not ivs:
            continue
        new_study.add_trial(
            create_trial(
                state=TrialState.COMPLETE,
                value=0.0,  # required by create_trial; base pruners use intermediate_values, not value
                params=t.params,
                distributions=t.distributions,
                intermediate_values=ivs,
            )
        )
    new_trial = create_trial(
        state=TrialState.RUNNING,
        params=trial.params,
        distributions=trial.distributions,
        intermediate_values=current_ivs,
    )
    return new_study, new_trial


def _create_single_metric_study_and_trial_multi(
    study: Study,
    trial: FrozenTrial,
    metric_directions: dict[str, StudyDirection],
) -> tuple[Study, FrozenTrial]:
    metric_names = list(metric_directions.keys())
    directions_list = list(metric_directions.values())

    all_trials = study.get_trials(deepcopy=False)
    trials = [t for t in all_trials if t.number != trial.number]

    def _extract(data: dict) -> dict[int, list[float]]:
        # Only include steps where all metrics are present.
        return {
            int(k): [v[m] for m in metric_names]
            for k, v in data.items()
            if all(m in v for m in metric_names)
        }

    current_data = _extract(trial.user_attrs.get(_USER_ATTR_KEY, {}))
    other_data_list = [_extract(t.user_attrs.get(_USER_ATTR_KEY, {})) for t in trials]

    current_ranks: dict[int, float] = {}
    other_ranks: list[dict[int, float]] = [{} for _ in trials]

    for step in current_data:
        entries: list[tuple[int, list[float]]] = [(-1, current_data[step])]
        for i, t_data in enumerate(other_data_list):
            if step in t_data:
                entries.append((i, t_data[step]))

        if len(entries) < 2:
            current_ranks[step] = 0.0
            continue

        ranks = _nondomination_rank([e[1] for e in entries], directions_list)
        for j, (idx, _) in enumerate(entries):
            if idx == -1:
                current_ranks[step] = float(ranks[j])
            else:
                other_ranks[idx][step] = float(ranks[j])

    return _build_synthetic_study_and_trial(
        trial, trials, other_ranks, current_ranks, direction="minimize"
    )


def _create_single_metric_study_and_trial_single(
    study: Study,
    trial: FrozenTrial,
    metric_name: str,
    direction: StudyDirection,
) -> tuple[Study, FrozenTrial]:
    trials = [t for t in study.get_trials(deepcopy=False) if t.number != trial.number]
    current_ivs: dict[int, float] = {
        int(k): v[metric_name]
        for k, v in trial.user_attrs.get(_USER_ATTR_KEY, {}).items()
        if metric_name in v
    }
    other_ivs = [
        {
            int(k): v[metric_name]
            for k, v in t.user_attrs.get(_USER_ATTR_KEY, {}).items()
            if metric_name in v
        }
        for t in trials
    ]
    return _build_synthetic_study_and_trial(
        trial, trials, other_ivs, current_ivs, direction=direction
    )


class MultiMetricPruner(BasePruner):
    """Pruner that supports multi-metric and named-metric intermediate value reporting.

    Optuna's native pruning mechanism does not support multi-objective optimization because
    :meth:`optuna.trial.Trial.report` raises :exc:`NotImplementedError` for multi-objective
    studies. This pruner overcomes that limitation by storing intermediate values in trial
    user attributes and building a synthetic single-objective study for the wrapped
    ``base_pruner`` to evaluate.

    Two reporting / pruning modes are selected via the ``joint`` argument:

    - **Multi-metric mode** (``joint=True``): Call :meth:`MultiMetricPrunerTrial.report`
      with a dict containing all metrics at each step, then call :meth:`should_prune` with
      no argument. The pruner converts the multi-dimensional intermediate values to Pareto
      ranks and passes those ranks to the base pruner.

    - **Per-metric mode** (``joint=False``): Call :meth:`MultiMetricPrunerTrial.report`
      with any number of metrics; each metric is handled independently. Calling
      :meth:`should_prune` with no argument checks all metrics and prunes if any of them
      individually triggers the base pruner. You can also call :meth:`should_prune` with
      ``metric_name`` to check a single metric.

    Args:
        base_pruner:
            An Optuna pruner to delegate the actual pruning decision to.
        metric_directions:
            A mapping from metric name to direction, e.g.
            ``{"loss": "minimize", "accuracy": "maximize"}``. All metrics reported via
            :meth:`MultiMetricPrunerTrial.report` must be keys in this mapping.
        joint:
            If :obj:`True`, all reported metrics are considered jointly using Pareto ranking
            (multi-metric mode). If :obj:`False`, each metric is evaluated independently by
            the base pruner (per-metric mode).

    Example:

        Multi-metric mode::

            import optuna
            import optunahub

            module = optunahub.load_local_module("pruners/multi_metric_pruner", registry_root="package/")
            MultiMetricPruner = module.MultiMetricPruner
            MultiMetricPrunerTrial = module.MultiMetricPrunerTrial

            def objective(trial):
                trial = MultiMetricPrunerTrial(trial)
                for step in range(10):
                    trial.report({"loss": metric1, "acc": metric2}, step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                return final_metric1, final_metric2

            study = optuna.create_study(
                directions=["minimize", "minimize"],
                pruner=MultiMetricPruner(
                    optuna.pruners.MedianPruner(),
                    metric_directions={"loss": "minimize", "acc": "minimize"},
                    joint=True,
                ),
            )
            study.optimize(objective, n_trials=20)
    """

    def __init__(
        self,
        base_pruner: BasePruner,
        *,
        metric_directions: dict[str, str | StudyDirection],
        joint: bool,
    ) -> None:
        direction_choices = [
            "minimize",
            "maximize",
            StudyDirection.MINIMIZE,
            StudyDirection.MAXIMIZE,
        ]
        if not metric_directions:
            raise ValueError("`metric_directions` must have at least one entry.")
        if any(d not in direction_choices for d in metric_directions.values()):
            raise ValueError(
                f"`metric_directions` values must be 'minimize' or 'maximize', "
                f"but got {metric_directions=}."
            )
        self._base_pruner = base_pruner
        self._joint = joint
        self._metric_directions = deepcopy(metric_directions)

    def prune(self, study: Study, trial: FrozenTrial, *, metric_name: str | None = None) -> bool:
        """Determine whether the trial should be pruned.

        Args:
            study: A study object.
            trial: A frozen trial object of the running trial.
            metric_name: Only used when ``joint=False``. If specified, prune based on this
                single named metric. If :obj:`None`, iterate over all metrics in
                ``metric_directions`` and prune if any of them triggers the base pruner.
                Ignored when ``joint=True``.

        Returns:
            :obj:`True` if the trial should be pruned.
        """
        if _USER_ATTR_KEY not in trial.user_attrs:
            return False

        if self._joint:
            new_study, new_trial = _create_single_metric_study_and_trial_multi(
                study, trial, self._metric_directions
            )
            return self._base_pruner.prune(new_study, new_trial)

        if metric_name is None or metric_name not in self._metric_directions:
            for name, direction in self._metric_directions.items():
                new_study, new_trial = _create_single_metric_study_and_trial_single(
                    study, trial, name, direction
                )
                if self._base_pruner.prune(new_study, new_trial):
                    return True
            return False

        if metric_name not in self._metric_directions:
            metric_directions = self._metric_directions
            raise ValueError(f"{metric_name=} is not in {metric_directions=}.")
        direction = self._metric_directions[metric_name]
        new_study, new_trial = _create_single_metric_study_and_trial_single(
            study, trial, metric_name, direction
        )
        return self._base_pruner.prune(new_study, new_trial)
