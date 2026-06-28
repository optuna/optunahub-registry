from __future__ import annotations

from typing import TYPE_CHECKING

from optuna._warnings import optuna_warn

from ._pruner import _USER_ATTR_KEY
from ._pruner import MultiMetricPruner


if TYPE_CHECKING:
    from typing import Any

    from optuna.trial import Trial


def _cast_value_to_float(value: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        msg = f"`values` must be a dict of float but got value with type {type(value)}."
        raise TypeError(msg) from None


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
                trial.report({"loss": metric1, "acc": metric2}, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return x**2, (x - 2.0) ** 2
    """

    def __init__(self, trial: Trial) -> None:
        self._trial = trial

    def __getattr__(self, name: str) -> Any:
        return getattr(self._trial, name)

    def report(self, values: dict[str, float], step: int) -> None:
        """Report intermediate metric values at a given step.

        Args:
            values: A dict mapping metric names to float values. All keys must be present
                in ``metric_directions``. Pass all metrics for multi-metric (Pareto) mode,
                or a single-entry dict for per-metric mode.
            step: Step of the trial (e.g., training epoch).
        """
        step = _cast_step_to_int(step)
        if not isinstance(values, dict):
            raise TypeError(f"`values` must be a dict but got {type(values)}.")
        if len(values) == 0:
            raise ValueError("`values` must have at least one entry.")

        pruner = self._trial.study.pruner
        if isinstance(pruner, MultiMetricPruner):
            unknown_keys = set(values.keys()) - set(pruner._metric_directions.keys())
            if unknown_keys:
                raise ValueError(f"Got unknown metric names `{unknown_keys}` in `values`.")
            if not pruner._joint and len(values) > 1:
                for k, v in values.items():
                    self.report({k: v}, step)
                return

        float_values = {k: _cast_value_to_float(v) for k, v in values.items()}
        data = dict(self._trial.user_attrs.get(_USER_ATTR_KEY, {}))
        # RDBStorages JSON-serialize user attrs, turning int keys into strings; use str upfront so
        # readers always see consistent key types.
        str_step = str(step) 
        step_data = dict(data.get(str_step, {}))
        already_reported = set(float_values) & set(step_data)
        if already_reported:
            optuna_warn(
                f"The reported values for {already_reported} are ignored because "
                f"already reported at step={step}."
            )
            float_values = {k: v for k, v in float_values.items() if k not in already_reported}
        if float_values:
            step_data.update(float_values)
            data[str_step] = step_data
            self._trial.set_user_attr(_USER_ATTR_KEY, data)

    def should_prune(self, *, metric_name: str | None = None) -> bool:
        """Check whether the trial should be pruned.

        The pruning mode is determined by the ``joint`` argument of :class:`MultiMetricPruner`.
        When ``joint=True``, all metrics are considered jointly via Pareto ranking and
        ``metric_name`` is ignored. When ``joint=False``, each metric is evaluated
        independently; passing ``metric_name`` restricts the check to that single metric,
        while omitting it checks all metrics and prunes if any triggers the base pruner.

        Args:
            metric_name: Only used when ``joint=False``. If specified, prune based on this
                single named metric. Ignored when ``joint=True``.

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
