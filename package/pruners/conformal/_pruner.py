"""Optuna integration: a pruner that runs in shadow mode first, then kills with a certified bound.

    pruner = ConformalPruner(n_calibration=60, alpha=0.05)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(), pruner=pruner)

**How it behaves.**
- **Shadow mode first.** The first `n_calibration` completed trials, by trial number, run
  unpruned. Their reported curves calibrate the rule once.
- **Then frozen.** The rule never recalibrates on later trials: those are survivors of the
  rule's own kills, and fitting on them would bias it.
- **The final outcome is the last reported intermediate value,** not `trial.value`.
- **Every calibration trial must report on the same step grid.**

**The guarantee needs the later trials to be exchangeable with the calibration trials.** With
`RandomSampler`, `QMCSampler` or a shuffled `GridSampler` they are. With `TPESampler`, `GPSampler`
or `CmaEsSampler` they are not, so the pruner warns once and the bound should not be relied on.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import optuna
from optuna.pruners import BasePruner
from optuna.study import StudyDirection
from optuna.trial import TrialState

from ._core import KillRule
from ._core import last_value
from ._core import Predictor


ArrayLike = Any

_EXCHANGEABLE = (
    optuna.samplers.RandomSampler,
    optuna.samplers.QMCSampler,
    optuna.samplers.GridSampler,
    optuna.samplers.BruteForceSampler,
)


class ConformalPruner(BasePruner):
    def __init__(
        self,
        n_calibration: int = 60,
        alpha: float = 0.05,
        *,
        good_threshold: float | None = None,
        good_quantile: float = 0.20,
        min_peek: int = 0,
        hold_until_resolved: float | None = None,
        predictor: Predictor = last_value,
    ) -> None:
        self.n_calibration = n_calibration
        self._kw: dict[str, Any] = dict(
            alpha=alpha,
            good_threshold=good_threshold,
            good_quantile=good_quantile,
            min_peek=min_peek,
            hold_until_resolved=hold_until_resolved,
            predictor=predictor,
        )
        self.rule_: KillRule | None = None
        self.steps_: list[int] | None = None
        self.calibration_trials_: list[int] = []
        self._warned = False

    @classmethod
    def from_curves(
        cls, curves: ArrayLike, steps: Any, *, maximize: bool = False, **kw: Any
    ) -> "ConformalPruner":
        """A pruner already calibrated offline, e.g. on a previous sweep's completed curves."""
        p = cls(n_calibration=len(curves), **kw)
        p.rule_ = KillRule(maximize=maximize, **p._kw).fit(curves)
        p.steps_ = list(steps)
        return p

    def _calibrate(self, study: optuna.study.Study) -> bool:
        done = sorted(
            (
                t
                for t in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
                if t.intermediate_values
            ),
            key=lambda t: t.number,
        )[: self.n_calibration]
        if len(done) < self.n_calibration:
            return False
        grid = sorted(done[0].intermediate_values)
        rows = [
            [t.intermediate_values[s] for s in grid]
            for t in done
            if sorted(t.intermediate_values) == grid
        ]
        if len(rows) < len(done):
            warnings.warn(
                f"{len(done) - len(rows)} calibration trials reported a different step "
                "grid and were skipped",
                RuntimeWarning,
                stacklevel=3,
            )
            return False
        maximize = study.direction == StudyDirection.MAXIMIZE
        self.rule_ = KillRule(maximize=maximize, **self._kw).fit(np.asarray(rows))
        self.steps_ = grid
        self.calibration_trials_ = [t.number for t in done]
        return True

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        if not self._warned and not isinstance(study.sampler, _EXCHANGEABLE):
            warnings.warn(
                f"{type(study.sampler).__name__} adapts to earlier trials, so later trials "
                "are not exchangeable with the calibration trials and the false-kill "
                "bound does not hold. Use RandomSampler or QMCSampler for a certified "
                "bound.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned = True
        step = trial.last_step
        if step is None:
            return False
        if self.rule_ is None and not self._calibrate(study):
            return False
        assert self.rule_ is not None and self.steps_ is not None
        if trial.number in self.calibration_trials_ or step not in self.steps_:
            return False
        idx = self.steps_.index(step)
        values = trial.intermediate_values
        prefix = [values.get(s) for s in self.steps_[: idx + 1]]
        if any(v is None for v in prefix):
            return False
        return self.rule_.should_kill(np.asarray(prefix, dtype=float))
