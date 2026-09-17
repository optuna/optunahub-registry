"""A kill rule for training runs that bounds how often it kills a run that would have ended well.

The construction is the one CC-03c pre-registered and measured on 400 real LoRA runs
(`experiments/2026-08-17-CC-03c-validity-gate-that-can-fire/`):

1. Calibrate on n **completed** runs, each a learning curve on a shared step grid.
2. A run is *good* if its final value is at or below a threshold τ (lower is better).
3. A predictor forecasts the final value from the curve so far, at every step.
4. Each good calibration run scores max over eligible steps of (prediction − τ).
   Runs that are not good score −∞.
5. λ̂ is the ⌈(1−α)(n+1)⌉-th smallest score. This is conformal risk control on the loss
   "this run is good AND the rule kills it".
6. A new run is killed at the first eligible step where prediction − λ̂ > τ.

**Guarantee.** If the calibration runs and the new runs are exchangeable — for example, drawn
i.i.d. by the same random sampler — then the expected fraction of new runs that are good and
killed is at most α. One kill decision covers every step at once, with no multiplicity
correction.

**What it does not promise.**
- **It bounds the joint rate, not the share of good runs killed.** With a small good fraction,
  the conditional rate can be large: ~30% on CC-03c.
- **It needs exchangeability.** An adaptive sampler such as TPE shifts later runs away from the
  calibration runs, and the guarantee no longer applies.
- **With τ estimated from the calibration finals** (`good_quantile`), the bound is measured, not
  proven: 0.88–0.97× α in CC-03c's simulations. Pass `good_threshold` when you have a real target.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
from typing import Any
from typing import Callable
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import rankdata


Predictor = Callable[[np.ndarray], np.ndarray]


def last_value(curves: ArrayLike) -> np.ndarray:
    """Forecast the final value as the latest observed value.

    A predictor maps curves of shape (n, T) to forecasts of shape (n, T) and must be causal:
    column t may use only columns 0..t. On CC-03c's pool this beat a fitted pow3 curve at every
    early step.
    """
    return np.asarray(curves, dtype=float)


class NotEnoughCalibration(ValueError):
    """Too few calibration runs for any threshold to certify at this α."""


def _spearman_cols(M: np.ndarray, y: np.ndarray) -> np.ndarray:
    rm = rankdata(M, axis=0)
    rm -= rm.mean(axis=0)
    ry = rankdata(y)
    ry -= ry.mean()
    denom = np.sqrt((rm**2).sum(axis=0) * (ry**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (rm * ry[:, None]).sum(axis=0) / denom, 0.0)


@dataclass
class KillRule:
    """Certified early-kill rule. Fit on completed curves, then ask `should_kill` on partial ones.

    Parameters
    ----------
    alpha : bound on the expected fraction of runs that are good and killed.
    good_threshold : a run is good if its final value is at or below this. Overrides `good_quantile`.
    good_quantile : if no threshold is given, τ is this quantile of the calibration finals.
    min_peek : index of the earliest step at which a kill may fire.
    hold_until_resolved : if set (e.g. 0.90), also hold kills until the first step at or after
        `min_peek` where Spearman(prediction, final) on the calibration runs reaches this value.
        Estimated per calibration set, so it varies between sets; pin `min_peek` instead when a
        prior pool tells you where the ranking settles.
    maximize : set True when higher values are better (accuracy, reward).
    predictor : causal forecaster of the final value. Default: last observed value.
    """

    alpha: float = 0.05
    good_threshold: float | None = None
    good_quantile: float = 0.20
    min_peek: int = 0
    hold_until_resolved: float | None = None
    maximize: bool = False
    predictor: Predictor = last_value

    tau_: float = field(init=False, default=math.nan)
    lambda_: float = field(init=False, default=math.nan)
    first_peek_: int = field(init=False, default=-1)
    n_steps_: int = field(init=False, default=0)
    n_calibration_: int = field(init=False, default=0)
    n_good_: int = field(init=False, default=0)
    resolution_: np.ndarray | None = field(init=False, default=None)

    # ── fitting ────────────────────────────────────────────────────────────────
    def _oriented(self, curves: ArrayLike) -> np.ndarray:
        Y = np.asarray(curves, dtype=float)
        return -Y if self.maximize else Y

    def fit(self, curves: ArrayLike) -> "KillRule":
        Y = self._oriented(curves)
        if Y.ndim != 2 or Y.shape[1] < 2:
            raise ValueError("curves must be a 2-D array (runs × steps) with at least 2 steps")
        if not np.all(np.isfinite(Y)):
            raise ValueError("calibration curves must be finite; drop incomplete runs first")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        n, T = Y.shape
        order = math.ceil((1 - self.alpha) * (n + 1))
        if order > n:
            raise NotEnoughCalibration(
                f"alpha={self.alpha} needs at least {math.ceil(1 / self.alpha) - 1} calibration "
                f"runs for any threshold to certify; got {n}"
            )

        finals = Y[:, -1]
        if self.good_threshold is not None:
            tau = -self.good_threshold if self.maximize else float(self.good_threshold)
        else:
            tau = float(np.sort(finals)[math.ceil(self.good_quantile * n) - 1])
        P = np.asarray(self.predictor(Y), dtype=float)
        if P.shape != Y.shape:
            raise ValueError(f"predictor returned shape {P.shape}, expected {Y.shape}")

        last_eligible = T - 2  # a kill at the final step saves nothing
        j0 = self.min_peek
        if self.hold_until_resolved is not None:
            rho = _spearman_cols(P[:, : T - 1], finals)
            self.resolution_ = rho
            hit = np.flatnonzero(rho[self.min_peek :] >= self.hold_until_resolved)
            j0 = self.min_peek + int(hit[0]) if hit.size else T - 1

        good = finals <= tau
        if j0 <= last_eligible:
            scores = np.where(good, P[:, j0 : last_eligible + 1].max(axis=1) - tau, -np.inf)
        else:
            scores = np.full(n, -np.inf)
        lam = float(np.sort(scores)[order - 1])
        if lam == -np.inf and j0 <= last_eligible:
            warnings.warn(
                f"only {int(good.sum())} good calibration runs: at alpha={self.alpha} no finite "
                "threshold certifies, so the rule kills every run at its first eligible step. "
                "Raise good_quantile or add calibration runs.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.tau_, self.lambda_, self.first_peek_ = tau, lam, j0
        self.n_steps_, self.n_calibration_, self.n_good_ = T, n, int(good.sum())
        return self

    # ── deciding ───────────────────────────────────────────────────────────────
    def _check_fitted(self) -> None:
        if self.n_steps_ == 0:
            raise RuntimeError("call fit() first")

    def should_kill(self, partial_curve: ArrayLike) -> bool:
        """True if a run whose curve so far is `partial_curve` should be killed at its latest step."""
        self._check_fitted()
        y = self._oriented(partial_curve)
        t = y.shape[0] - 1
        if t < self.first_peek_ or t > self.n_steps_ - 2:
            return False
        pred = float(np.asarray(self.predictor(y[None, :]))[0, t])
        return bool(pred - self.lambda_ > self.tau_)

    def kill_step(self, curve: ArrayLike) -> int | None:
        """Index of the first step at which the rule would kill this curve, or None."""
        self._check_fitted()
        Y = self._oriented(curve)[None, :]
        P = np.asarray(self.predictor(Y), dtype=float)[0]
        lo, hi = self.first_peek_, min(self.n_steps_, Y.shape[1]) - 2
        if lo > hi:
            return None
        fire = np.flatnonzero(P[lo : hi + 1] - self.lambda_ > self.tau_)
        return lo + int(fire[0]) if fire.size else None

    def evaluate(self, curves: ArrayLike) -> dict[str, Any]:
        """Score the rule on held-out COMPLETED curves: false kills, joint rate, savings."""
        self._check_fitted()
        Y = self._oriented(curves)
        n, T = Y.shape
        P = np.asarray(self.predictor(Y), dtype=float)
        lo, hi = self.first_peek_, T - 2
        if lo <= hi:
            fire = P[:, lo : hi + 1] - self.lambda_ > self.tau_
            killed = fire.any(axis=1)
            step = np.where(killed, lo + np.argmax(fire, axis=1), T - 1)
        else:
            killed, step = np.zeros(n, bool), np.full(n, T - 1)
        good = Y[:, -1] <= self.tau_
        fk = int((killed & good).sum())
        return {
            "n": n,
            "n_good": int(good.sum()),
            "n_killed": int(killed.sum()),
            "false_kills": fk,
            "joint_false_kill_rate": fk / n,
            "conditional_false_kill_rate": fk / int(good.sum()) if good.any() else None,
            "savings": float(1 - (step + 1).sum() / (T * n)),
            "median_kill_step": float(np.median(step[killed])) if killed.any() else None,
        }
