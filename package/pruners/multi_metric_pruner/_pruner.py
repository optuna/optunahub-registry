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
    from collections.abc import Sequence

    from optuna.study import Study
    from optuna.trial import FrozenTrial


_USER_ATTR_KEY_MULTI = "multi_pruner:multi:"
_USER_ATTR_KEY_PREFIX_SINGLE = "multi_pruner:single:"


def _nondomination_rank(
    values_list: list[list[float]], directions: list[StudyDirection] | None
) -> np.ndarray:
    lvals = np.array(values_list, dtype=float)
    if directions is not None:
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


def _create_single_metric_study_and_trial_multi(
    study: Study,
    trial: FrozenTrial,
    directions: list[StudyDirection] | None,
) -> tuple[Study, FrozenTrial]:
    all_trials = study.get_trials(deepcopy=False)
    other_trials = [t for t in all_trials if t.number != trial.number]

    current_data = {int(k): v for k, v in trial.user_attrs.get(_USER_ATTR_KEY_MULTI, {}).items()}
    other_data_list = [
        {int(k): v for k, v in t.user_attrs.get(_USER_ATTR_KEY_MULTI, {}).items()}
        for t in other_trials
    ]

    current_ranks: dict[int, float] = {}
    other_ranks: list[dict[int, float]] = [{} for _ in other_trials]

    for step in current_data:
        entries: list[tuple[int, list[float]]] = []
        entries.append((-1, current_data[step]))
        for i, t_data in enumerate(other_data_list):
            if step in t_data:
                entries.append((i, t_data[step]))

        if len(entries) < 2:
            # Only the current trial has reported at this step; rank is 0.
            current_ranks[step] = 0.0
            continue

        values_list = [e[1] for e in entries]
        ranks = _nondomination_rank(values_list, directions)

        for j, (idx, _) in enumerate(entries):
            if idx == -1:
                current_ranks[step] = float(ranks[j])
            else:
                other_ranks[idx][step] = float(ranks[j])

    with _suppress_new_study_log():
        new_study = optuna.create_study(direction="minimize")

    for i, t in enumerate(other_trials):
        if not t.state.is_finished():
            continue
        ivs = other_ranks[i]
        if not ivs:
            continue
        last_rank = float(ivs[max(ivs.keys())])
        new_study.add_trial(
            create_trial(
                state=TrialState.COMPLETE,
                value=last_rank,
                params=t.params,
                distributions=t.distributions,
                intermediate_values=ivs,
            )
        )

    new_trial = create_trial(
        state=TrialState.RUNNING,
        params=trial.params,
        distributions=trial.distributions,
        intermediate_values=current_ranks,
    )
    return new_study, new_trial


def _create_single_metric_study_and_trial_single(
    study: Study,
    trial: FrozenTrial,
    metric_name: str,
    direction: StudyDirection,
) -> tuple[Study, FrozenTrial]:
    key = f"{_USER_ATTR_KEY_PREFIX_SINGLE}{metric_name}"
    all_trials = study.get_trials(deepcopy=False)
    other_trials = [t for t in all_trials if t.number != trial.number]

    current_ivs: dict[int, float] = {int(k): v for k, v in trial.user_attrs.get(key, {}).items()}
    direction_str = "maximize" if direction == StudyDirection.MAXIMIZE else "minimize"
    with _suppress_new_study_log():
        new_study = optuna.create_study(direction=direction_str)

    for t in other_trials:
        if not t.state.is_finished():
            continue
        t_ivs: dict[int, float] = {int(k): v for k, v in t.user_attrs.get(key, {}).items()}
        if not t_ivs:
            continue
        new_study.add_trial(
            create_trial(
                state=TrialState.COMPLETE,
                value=0.0,
                params=t.params,
                distributions=t.distributions,
                intermediate_values=t_ivs,
            )
        )

    new_trial = create_trial(
        state=TrialState.RUNNING,
        params=trial.params,
        distributions=trial.distributions,
        intermediate_values=current_ivs,
    )
    return new_study, new_trial


class MultiMetricPruner(BasePruner):
    """Pruner that supports multi-metric and named-metric intermediate value reporting.

    Optuna's native pruning mechanism does not support multi-objective optimization because
    :meth:`optuna.trial.Trial.report` raises :exc:`NotImplementedError` for multi-objective
    studies. This pruner overcomes that limitation by storing intermediate values in trial
    user attributes and building a synthetic single-objective study for the wrapped
    ``base_pruner`` to evaluate.

    Two reporting modes are supported (choose one per trial, do not mix them):

    - **Multi-metric mode**: Call :meth:`MultiMetricPrunerTrial.report` with a sequence of
      floats. The pruner converts the multi-dimensional intermediate values to Pareto ranks
      and passes those ranks to the base pruner as single-metric values. Use
      :meth:`prune` with ``metric_name=None``.

    - **Single-metric mode**: Call :meth:`MultiMetricPrunerTrial.report` with a float and a
      ``metric_name``. The pruner extracts the specified metric's values and passes them
      to the base pruner. Use :meth:`prune` with ``metric_name`` set to the target metric.

    Args:
        base_pruner:
            An Optuna pruner to delegate the actual pruning decision to.
        directions:
            Directions for each metric in multi-metric mode (e.g.,
            ``["minimize", "maximize"]``). Defaults to minimization for all metrics when
            :obj:`None`.
        metric_directions:
            A mapping from metric name to direction (e.g.,
            ``{"loss": "minimize", "accuracy": "maximize"}``) for single-metric mode. A
            metric not listed here defaults to ``"minimize"``.

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
                    trial.report([metric1, metric2], step)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                return final_metric1, final_metric2

            study = optuna.create_study(
                directions=["minimize", "minimize"],
                pruner=MultiMetricPruner(optuna.pruners.MedianPruner()),
            )
            study.optimize(objective, n_trials=20)
    """

    def __init__(
        self,
        base_pruner: BasePruner,
        *,
        directions: Sequence[str | StudyDirection] | None = None,
        metric_directions: dict[str, str | StudyDirection] | None = None,
    ) -> None:
        direction_choices = [
            "minimize",
            "maximize",
            StudyDirection.MINIMIZE,
            StudyDirection.MAXIMIZE,
        ]
        if directions is None and metric_directions is None:
            raise ValueError("Specify either `directions` or `metric_directions`.")
        if directions is not None and metric_directions is not None:
            raise ValueError("Specify only one of `directions` and `metric_directions`.")
        if directions is not None and any(d not in direction_choices for d in directions):
            raise ValueError(
                f"`directions` must be a list of `minimize` or `maximize`, but got {directions=}."
            )
        if metric_directions is not None and any(
            d not in direction_choices for d in metric_directions.values()
        ):
            raise ValueError(
                f"`metric_directions` must be a list of `minimize` or `maximize`, but got {metric_directions=}."
            )
        self._base_pruner = base_pruner
        self._directions: list[StudyDirection] | None = deepcopy(directions)
        self._metric_directions: dict[str, StudyDirection] = deepcopy(metric_directions)

    def prune(self, study: Study, trial: FrozenTrial, *, metric_name: str | None = None) -> bool:
        """Determine whether the trial should be pruned.

        Args:
            study: A study object.
            trial: A frozen trial object of the running trial.
            metric_name: If specified, prune based on this named metric (reported via
                :meth:`MultiMetricPrunerTrial.report` with a ``metric_name``). If
                :obj:`None`, prune based on jointly reported values (reported via
                :meth:`MultiMetricPrunerTrial.report` with a sequence of floats).

        Returns:
            :obj:`True` if the trial should be pruned.

        Raises:
            ValueError: If the reported values are incompatible with ``metric_name``.
        """
        has_multi = _USER_ATTR_KEY_MULTI in trial.user_attrs
        has_single = any(k.startswith(_USER_ATTR_KEY_PREFIX_SINGLE) for k in trial.user_attrs)

        if has_multi and has_single:
            raise ValueError(
                "Both multi-metric and single-metric values were reported in the same trial. "
                "Use only one reporting mode."
            )

        if metric_name is None:
            if has_single:
                raise ValueError(
                    "Single-metric values were reported but metric_name is not specified in "
                    "prune(). Pass metric_name to prune() or report a sequence of floats instead."
                )
            if not has_multi:
                return False
            new_study, new_trial = _create_single_metric_study_and_trial_multi(
                study, trial, self._directions
            )
        else:
            if has_multi:
                raise ValueError(
                    f"Multi-metric values were reported but metric_name={metric_name!r} was "
                    "passed to prune(). Report a sequence of floats and use metric_name=None, "
                    "or report a float with a metric_name."
                )
            key = f"{_USER_ATTR_KEY_PREFIX_SINGLE}{metric_name}"
            if key not in trial.user_attrs:
                return False
            direction = self._metric_directions.get(metric_name, StudyDirection.MINIMIZE)
            new_study, new_trial = _create_single_metric_study_and_trial_single(
                study, trial, metric_name, direction
            )

        return self._base_pruner.prune(new_study, new_trial)
