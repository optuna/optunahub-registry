"""Optuna sampler for distributionally robust level-set estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.distributions import IntDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub

from ._acquisition import acquisition_values
from ._lse import DRLevelSetEstimator
from ._lse import gaussian_kernel


class DRLevelSetSampler(optunahub.samplers.SimpleBaseSampler):
    """Active learning for the distributionally robust reliable set.

    The objective is a black box ``f(x, w)`` of a controllable design variable ``x`` and an
    uncontrollable environmental variable ``w``. Both range over finite grids supplied up
    front. The sampler chooses which ``(x, w)`` pair to evaluate next so as to identify the
    reliable set ``H = {x : F(x) > alpha}`` in as few evaluations as possible, where ``F`` is
    the distributionally robust probability that ``f(x, w)`` exceeds ``h``.

    Unlike an optimisation sampler, the result of a study is not ``study.best_trial`` but the
    classification returned by :meth:`classify`. The study direction is ignored.

    Args:
        x_points: One-dimensional array of candidate design values.
        w_points: One-dimensional array of environmental values.
        h: Threshold defining the robustness event ``f(x, w) > h``.
        alpha: Level defining the reliable set.
        eps: Radius of the L1 ambiguity set around ``p_ref``.
        p_ref: Reference p.m.f. over ``w_points``. Defaults to uniform.
        eta: Accuracy parameter. Zero cannot misclassify given valid credible intervals but
            may never terminate; positive values terminate at a bounded loss.
        beta: Squared credible-interval width, ``mean +/- sqrt(beta) * sd``.
        gamma: Weight of the variance tie-break in the acquisition.
        zeta: Accuracy of the Lemma 3.3 region approximation. Smaller is more exact and
            slower; zero evaluates every region.
        sigma_f: Prior standard deviation of the process.
        scale: Divisor of the squared distance in the kernel; the square of a length scale.
        noise: Observation noise variance.
        n_startup_trials: Number of uniformly random evaluations before the acquisition is
            used.
        seed: Random seed.

    Reference:
        Yu Inatsu, Shogo Iwazaki and Ichiro Takeuchi. Active Learning for Distributionally
        Robust Level-Set Estimation. ICML 2021, PMLR 139:4574-4584.
    """

    def __init__(
        self,
        x_points: np.ndarray,
        w_points: np.ndarray,
        *,
        h: float,
        alpha: float,
        eps: float,
        p_ref: np.ndarray | None = None,
        eta: float = 0.0,
        beta: float = 4.0,
        gamma: float = 1e-3,
        zeta: float = 1e-6,
        sigma_f: float = 1.0,
        scale: float = 1.0,
        noise: float = 1e-4,
        n_startup_trials: int = 5,
        seed: int | None = None,
    ) -> None:
        self._x_points = np.asarray(x_points, dtype=float)
        self._w_points = np.asarray(w_points, dtype=float)
        self._n_x = len(self._x_points)
        self._n_w = len(self._w_points)
        if self._n_x < 1 or self._n_w < 1:
            raise ValueError("x_points and w_points must each contain at least one value.")
        if n_startup_trials < 1:
            raise ValueError(f"n_startup_trials must be positive, got {n_startup_trials}.")

        kernel, _ = gaussian_kernel(self._x_points, self._w_points, sigma_f, scale)
        self._kernel = kernel
        self._p_ref = (
            np.full(self._n_w, 1.0 / self._n_w) if p_ref is None else np.asarray(p_ref, float)
        )
        self._h, self._alpha, self._eps = h, alpha, eps
        self._eta, self._beta, self._gamma, self._noise = eta, beta, gamma, noise
        self._zeta = zeta
        self._n_startup_trials = n_startup_trials
        self._rng = np.random.default_rng(seed)
        # Validate the configuration eagerly rather than on the first trial.
        self._new_estimator()

        super().__init__(
            search_space={
                "x_index": IntDistribution(0, self._n_x - 1),
                "w_index": IntDistribution(0, self._n_w - 1),
            },
            seed=seed,
        )

    def _new_estimator(self) -> DRLevelSetEstimator:
        return DRLevelSetEstimator(
            self._kernel,
            self._n_x,
            self._n_w,
            self._p_ref,
            self._eps,
            self._h,
            self._alpha,
            eta=self._eta,
            noise=self._noise,
        )

    def _fit(self, study: Study) -> DRLevelSetEstimator:
        """Rebuild the estimator from the study's completed trials."""
        estimator = self._new_estimator()
        for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
            if trial.value is None:
                continue
            ix, iw = trial.params.get("x_index"), trial.params.get("w_index")
            if ix is None or iw is None:
                continue
            estimator.observe(int(ix), int(iw), float(trial.value))
        return estimator

    def classify(self, study: Study) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return boolean masks ``(H, L, U)`` over ``x_points`` given the study so far."""
        return self._fit(study).classify(self._beta)

    def reliable_set(self, study: Study) -> np.ndarray:
        """Return the design values currently classified as reliable."""
        reliable, _, _ = self.classify(study)
        return self._x_points[reliable]

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        estimator = self._fit(study)
        evaluated = set(estimator.observed_indices)

        def as_params(flat: int) -> dict[str, Any]:
            return {"x_index": flat // self._n_w, "w_index": flat % self._n_w}

        def random_unevaluated() -> dict[str, Any]:
            free = np.setdiff1d(np.arange(self._n_x * self._n_w), list(evaluated))
            if free.size == 0:
                return as_params(int(self._rng.integers(self._n_x * self._n_w)))
            return as_params(int(self._rng.choice(free)))

        if estimator.n_observations < self._n_startup_trials:
            return random_unevaluated()

        values, unclassified = acquisition_values(estimator, self._beta, gamma=self._gamma)
        if unclassified.size == 0:
            # Every design point is classified; the task is complete. Keep the study
            # runnable by continuing to explore rather than repeating a single point.
            return random_unevaluated()
        if evaluated:
            values[np.asarray(sorted(evaluated))] = -np.inf
        best = np.flatnonzero(values >= values.max() - 1e-12)
        return as_params(int(self._rng.choice(best)))
