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

from ._hypervolume._nondomination import _fast_non_domination_rank
from ._hypervolume._ordering import _argsort_by_hv_contribution


if TYPE_CHECKING:
    from collections.abc import Generator

    from optuna.study import Study
    from optuna.trial import FrozenTrial


_USER_ATTR_KEY = "multi_pruner:values"
# Example of the attribute data:
# {
#     "0": {"loss": 0.52, "acc": 0.81},  # The reported values at step 0.
#     "5": {"loss": 0.31},        # mixed-frequency: only one metric at this step
# }


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
    reported_values_list: list[dict[int, float]],
    current_reported_values: dict[int, float],
    *,
    direction: str | StudyDirection,
) -> tuple[Study, FrozenTrial]:
    with _suppress_new_study_log():
        new_study = optuna.create_study(direction=direction)
    for t, reported_values_list in zip(trials, reported_values_list):
        if not t.state.is_finished() or not reported_values_list:
            continue
        new_study.add_trial(
            create_trial(
                state=TrialState.COMPLETE,
                # required by create_trial; base pruners use intermediate_values, not value.
                value=0.0,
                params=t.params,
                distributions=t.distributions,
                intermediate_values=reported_values_list,
            )
        )
    new_trial = create_trial(
        state=TrialState.RUNNING,
        params=trial.params,
        distributions=trial.distributions,
        intermediate_values=current_reported_values,
    )
    return new_study, new_trial


def _tie_break(loss_values: np.ndarray, ranks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Break ties among trials sharing the current trial's rank by HV contribution.

    Returns ``(indices, bonuses)`` where ``bonuses`` slopes linearly from ``-0.5`` (largest HV
    contribution) to ``-0.1`` (smallest). The magnitude stays below ``1.0`` so every trial keeps
    its rank band.
    """
    is_same_rank_current_trial = ranks == ranks[-1]
    same_rank_indices = np.arange(len(loss_values))[is_same_rank_current_trial]
    if len(same_rank_indices) == 1:
        return same_rank_indices, np.zeros(1)

    rank_i_lvals = loss_values[is_same_rank_current_trial]
    worst_lvals = np.max(rank_i_lvals, axis=0)
    ref_point = np.maximum(1.1 * worst_lvals, 0.9 * worst_lvals)
    order = _argsort_by_hv_contribution(rank_i_lvals, ref_point)
    bonuses = -np.linspace(0.5, 0.1, len(order))
    return same_rank_indices[order], bonuses


def _create_single_metric_study_and_trial_multi(
    study: Study, trial: FrozenTrial, metric_directions: dict[str, StudyDirection]
) -> tuple[Study, FrozenTrial]:
    metric_names = list(metric_directions.keys())
    all_trials = study.get_trials(deepcopy=False)
    trials = [t for t in all_trials if t.number != trial.number]

    def _extract(data: dict) -> dict[int, list[float]]:
        # Only include steps where all metrics are present.
        return {
            int(k): [v[m] for m in metric_names]
            for k, v in data.items()
            if all(m in v for m in metric_names)
        }

    signs = np.array(
        [-1.0 if d == StudyDirection.MAXIMIZE else 1.0 for d in metric_directions.values()],
        dtype=float,
    )
    current_reported_values = _extract(trial.user_attrs.get(_USER_ATTR_KEY, {}))
    reported_values_list = [_extract(t.user_attrs.get(_USER_ATTR_KEY, {})) for t in trials]
    current_ranks: dict[int, float] = {}
    ranks_list: list[dict[int, float]] = [{} for _ in trials]
    for step in current_reported_values:
        trial_indices = [
            i for i, reported_vs in enumerate(reported_values_list) if step in reported_vs
        ]
        vs = [reported_vs[step] for reported_vs in reported_values_list if step in reported_vs]
        vs.append(current_reported_values[step])
        values_in_step = np.asarray(vs)
        if len(values_in_step) < 2:
            current_ranks[step] = 0.0
            continue

        loss_values = values_in_step * signs
        ranks = _fast_non_domination_rank(loss_values)
        rewards = ranks.astype(float)  # lower is better.
        if ranks[-1] == np.min(ranks):
            # If rank is zero, it means the trial is on the Pareto front. To prevent it from being
            # pruned unintentionally, we use -1.0. This ensures that this trial won't be pruned.
            rewards[-1] -= 1.0
        else:  # Tie-break non-Pareto solutions by HV contribution within the shared rank.
            tie_indices, tie_bonuses = _tie_break(loss_values, ranks)
            rewards[tie_indices] += tie_bonuses

        current_ranks[step] = rewards[-1].item()
        for i, trial_idx in enumerate(trial_indices):
            ranks_list[trial_idx][step] = rewards[i].item()

    return _build_synthetic_study_and_trial(
        trial, trials, ranks_list, current_ranks, direction="minimize"
    )


def _create_single_metric_study_and_trial_single(
    study: Study, trial: FrozenTrial, metric_name: str, direction: StudyDirection
) -> tuple[Study, FrozenTrial]:
    trials = [t for t in study.get_trials(deepcopy=False) if t.number != trial.number]
    current_reported_values: dict[int, float] = {
        int(k): v[metric_name]
        for k, v in trial.user_attrs.get(_USER_ATTR_KEY, {}).items()
        if metric_name in v
    }
    reported_values_list = [
        {
            int(k): v[metric_name]
            for k, v in t.user_attrs.get(_USER_ATTR_KEY, {}).items()
            if metric_name in v
        }
        for t in trials
    ]
    return _build_synthetic_study_and_trial(
        trial, trials, reported_values_list, current_reported_values, direction=direction
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

        if metric_name is not None and metric_name not in self._metric_directions:
            metric_directions = self._metric_directions
            raise ValueError(f"{metric_name=} is not in {metric_directions=}.")

        if metric_name is None:
            for name, direction in self._metric_directions.items():
                new_study, new_trial = _create_single_metric_study_and_trial_single(
                    study, trial, name, direction
                )
                if self._base_pruner.prune(new_study, new_trial):
                    return True
            return False
        direction = self._metric_directions[metric_name]
        new_study, new_trial = _create_single_metric_study_and_trial_single(
            study, trial, metric_name, direction
        )
        return self._base_pruner.prune(new_study, new_trial)
