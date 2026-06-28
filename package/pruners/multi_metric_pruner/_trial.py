from __future__ import annotations

from typing import TYPE_CHECKING

from optuna._warnings import optuna_warn

from ._pruner import _USER_ATTR_KEY_MULTI
from ._pruner import _USER_ATTR_KEY_PREFIX_SINGLE
from ._pruner import MultiMetricPruner


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from optuna.trial import Trial


def _cast_value_to_float(value: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        message = f"`values` must be float or a list of float but got {type(value)}."
        raise TypeError(message) from None


def _cast_step_to_int(step: int) -> int:
    if step < 0:
        raise ValueError(f"{step=} cannot be negative.")
    try:
        return int(step)
    except (TypeError, ValueError):
        raise TypeError(f"`step` must be int but got {type(step)}.") from None


class MultiMetricPrunerTrial:
    """A :class:`~optuna.trial.Trial` wrapper for use with :class:`MultiMetricPruner`.

    Wraps a trial to provide :meth:`report` and :meth:`should_prune` that work in
    multi-objective studies. All other trial attributes and methods are forwarded to the
    wrapped trial transparently via ``__getattr__``.

    Args:
        trial: The trial object received in the objective function.

    Example::

        def objective(trial: optuna.Trial) -> tuple[float, float]:
            trial = MultiMetricPrunerTrial(trial)
            x = trial.suggest_float("x", -5.0, 5.0)
            for step in range(10):
                trial.report([metric1, metric2], step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return x**2, (x - 2.0) ** 2
    """

    def __init__(self, trial: Trial) -> None:
        self._trial = trial

    def __getattr__(self, name: str) -> Any:
        return getattr(self._trial, name)

    def report(
        self,
        values: float | Sequence[float],
        step: int,
        *,
        metric_name: str | None = None,
    ) -> None:
        """Report intermediate metric value(s) at a given step.

        Args:
            values: A single float (single-metric mode) or a sequence of floats
                (multi-metric mode).
            step: Step of the trial (e.g., training epoch).
            metric_name: Required when ``values`` is a single float; the name of the
                metric to report. Must be :obj:`None` when ``values`` is a sequence.
        """
        step = _cast_step_to_int(step)
        if isinstance(values, float):
            if metric_name is None:
                raise ValueError("When `values` is float, metric_name must be specified.")
            key = f"{_USER_ATTR_KEY_PREFIX_SINGLE}{metric_name}"
            values = _cast_value_to_float(values)
        else:
            if metric_name is not None:
                raise ValueError("When `values` is not float, metric_name cannot be specified.")
            key = _USER_ATTR_KEY_MULTI
            values = [_cast_value_to_float(v) for v in values]

        if len(data := self._trial.user_attrs.get(key, {})) == 0:
            optuna_warn(
                f"The reported value is ignored because this `{step=}` is already reported."
            )
            return
        data[step] = values
        self._trial.set_user_attr(key, data)

    def should_prune(self, *, metric_name: str | None = None) -> bool:
        """Check whether the trial should be pruned.

        Args:
            metric_name: If specified, prune based on this named metric (single-metric
                mode). If :obj:`None`, prune based on jointly reported values
                (multi-metric mode).

        Returns:
            :obj:`True` if the trial should be pruned.

        Raises:
            ValueError: If the study's pruner is not a :class:`MultiMetricPruner`.
        """
        pruner = self._trial.study.pruner
        if not isinstance(pruner, MultiMetricPruner):
            pruner_name = pruner.__class__.__name__
            raise ValueError(
                f"MultiMetricPrunerTrial.should_prune() requires the study to use "
                f"MultiMetricPruner as its pruner, but got {pruner_name}."
            )
        frozen = self._trial._get_latest_trial()
        return pruner.prune(self._trial.study, frozen, metric_name=metric_name)
