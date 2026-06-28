from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence

    from optuna.trial import Trial


_USER_ATTR_KEY_MULTI = "multi_pruner:multi:"
_USER_ATTR_KEY_PREFIX_SINGLE = "multi_pruner:single:"


def trial_report_multi(trial: Trial, values: Sequence[float], step: int) -> None:
    """Report multiple intermediate metric values jointly at a given step.

    Cannot be combined with :func:`trial_report` in the same trial.
    Use :func:`MultiMetricPruner.prune` with ``metric_name=None`` to prune based on
    these jointly reported values via Pareto ranking.

    Args:
        trial: A trial object.
        values: A sequence of metric values to report.
        step: Step of the trial (e.g., training epoch).
    """
    data = trial.user_attrs.get(_USER_ATTR_KEY_MULTI, {})
    data[str(step)] = list(float(v) for v in values)
    trial.set_user_attr(_USER_ATTR_KEY_MULTI, data)


def trial_report(trial: Trial, value: float, step: int, *, metric_name: str) -> None:
    """Report a single named intermediate metric value at a given step.

    Cannot be combined with :func:`trial_report_multi` in the same trial.
    Use :func:`MultiMetricPruner.prune` with ``metric_name`` to prune based on this metric.

    Args:
        trial: A trial object.
        value: A metric value to report.
        step: Step of the trial (e.g., training epoch).
        metric_name: Name of the metric.
    """
    key = f"{_USER_ATTR_KEY_PREFIX_SINGLE}{metric_name}"
    data = trial.user_attrs.get(key, {})
    data[str(step)] = float(value)
    trial.set_user_attr(key, data)


def should_prune(trial: Trial, *, metric_name: str | None = None) -> bool:
    """Check whether the trial should be pruned.

    This is the multi-objective counterpart of :meth:`optuna.trial.Trial.should_prune`.

    Args:
        trial: A trial object.
        metric_name: If specified, prune based on this named metric (reported via
            :func:`trial_report`). If :obj:`None`, prune based on jointly reported values
            (reported via :func:`trial_report_multi`).

    Returns:
        :obj:`True` if the trial should be pruned.
    """
    frozen = trial._get_latest_trial()
    return trial.study.pruner.prune(trial.study, frozen, metric_name=metric_name)
