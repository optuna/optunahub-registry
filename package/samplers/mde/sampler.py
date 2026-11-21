"""Multi-population based Differential Evolution (MDE) sampler for Optuna.

Karkinli, A.E. (2023). Detection of object boundary from point cloud by using
multi-population based differential evolution algorithm.
Neural Computing and Applications, 35, 5193-5206.
https://doi.org/10.1007/s00521-022-07969-w

Original MATLAB implementation:
https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import math
import threading
from typing import Any
import warnings

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub


# ---------------------------------------------------------------------------
# MDE variation operators (port of ``algo_MDE.m``).
# ---------------------------------------------------------------------------


def _mde_scale(rng: np.random.Generator) -> float:
    """scale = |a - rand^b| * randn^d, a ~ U{0, 1}, b ~ U{1..10}, d ~ U{1..5}."""
    return float(
        abs(rng.integers(0, 2) - rng.random() ** rng.integers(1, 11))
        * rng.standard_normal() ** rng.integers(1, 6)
    )


def _mde_trial(
    B: np.ndarray, best: np.ndarray, noise: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Build the N trial vectors from the sub-population ``B``."""
    N, D = B.shape
    temp = np.empty_like(B)
    for i in range(N):
        scale = _mde_scale(rng)
        while True:  # r = permute(N, 2) with r(1) != i and r(2) != i.
            r1, r2 = rng.choice(N, 2, replace=False)
            if r1 != i and r2 != i:
                break
        dx = np.where(rng.random(D) < 0.5, B[r1], best)
        dy = np.where(rng.random(D) < 0.5, B[i], B[r2])
        temp[i] = B[r2] + scale * (dx - dy)
    # Parameter-free crossover control matrix.
    c = 1 if rng.random() ** rng.integers(1, 6) < 0.5 else D
    cmap = (
        np.abs(rng.integers(0, 2, (N, c)) - rng.random((N, D)) ** rng.integers(1, 6, (N, c))) < 0.5
    )
    return B + cmap * (temp + noise - B)


def _mde_noise(B: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """noise = B .* 10^phi .* (rand(N, 1) - 0.5), phi ~ U{-12..-9}."""
    N, D = B.shape
    return B * 10.0 ** rng.integers(-12, -8, (N, D)) * (rng.random((N, 1)) - 0.5)


def _mde_bound(
    X: np.ndarray, lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Re-sample out-of-bounds variables uniformly inside [lb, ub] (Eq. 15)."""
    out = (X < lb) | (X > ub)
    if out.any():
        R = lb + rng.random(X.shape) * (ub - lb)
        X = np.where(out, R, X)
    return X


# ---------------------------------------------------------------------------
# Search-space transform (numerical parameters only).
# ---------------------------------------------------------------------------


def _is_numerical(dist: BaseDistribution) -> bool:
    return isinstance(dist, (FloatDistribution, IntDistribution)) and not dist.single()


def _internal_bounds(dist: BaseDistribution) -> tuple[float, float]:
    assert isinstance(dist, (FloatDistribution, IntDistribution))
    low, high = float(dist.low), float(dist.high)
    if dist.log:
        if isinstance(dist, IntDistribution):
            return math.log(low - 0.5), math.log(high + 0.5)
        return math.log(low), math.log(high)
    if dist.step is not None:
        half = 0.5 * float(dist.step)
        return low - half, high + half
    return low, high


def _to_internal(dist: BaseDistribution, value: Any) -> float:
    assert isinstance(dist, (FloatDistribution, IntDistribution))
    return math.log(float(value)) if dist.log else float(value)


def _to_external(dist: BaseDistribution, x: float) -> float | int:
    assert isinstance(dist, (FloatDistribution, IntDistribution))
    v = math.exp(x) if dist.log else x
    if isinstance(dist, IntDistribution):
        step = 1 if dist.log else dist.step
        v = dist.low + round((v - dist.low) / step) * step
        return int(min(max(v, dist.low), dist.high))
    if dist.step is not None:
        v = dist.low + round((v - dist.low) / dist.step) * dist.step
    return float(min(max(v, dist.low), dist.high))


# ---------------------------------------------------------------------------
# Sampler.
# ---------------------------------------------------------------------------


class MDESampler(optunahub.samplers.SimpleBaseSampler):
    """Multi-population based Differential Evolution (MDE) sampler.

    MDE keeps a full population ``A`` of ``3 * population_size`` vectors. At every
    generation a sub-population ``B`` of ``population_size`` randomly chosen members of
    ``A`` is evolved with a parameter-free mutation/crossover scheme, the trial vectors are
    evaluated, better trial vectors replace their parents (greedy selection) and ``B`` is
    written back into ``A``.

    Mapping onto Optuna trials:

    - Generation 0 consists of ``3 * population_size`` trials that evaluate the uniformly
      initialized full population ``A``.
    - Every following generation consists of ``population_size`` trials, one per trial
      vector of ``B``.
    - When all trials of a generation have finished, the population is updated and the
      trial vectors of the next generation are created.

    Float parameters are handled directly; integer and stepped float parameters are
    optimized in a continuous relaxation and rounded, and log-scale parameters are
    optimized in log space. Categorical parameters (and any parameter outside the relative
    search space) are sampled by :class:`~optuna.samplers.RandomSampler`. Pruned and failed
    trials are treated as having an objective value of ``+inf`` (they never replace a
    population member).

    Args:
        search_space:
            Optional search space. If ``None``, the search space is inferred from the
            completed trials (intersection search space); in that case the very first
            trial is sampled randomly, before the MDE population is initialized.
        population_size:
            ``N``, the size of the sub-population evolved in every generation. The full
            population has ``3 * N`` members. Must be at least 3.
        seed:
            Seed for the random number generators.
    """

    def __init__(
        self,
        search_space: dict[str, BaseDistribution] | None = None,
        population_size: int = 30,
        seed: int | None = None,
    ) -> None:
        if population_size < 3:
            raise ValueError("`population_size` must be >= 3.")
        super().__init__(search_space, seed)
        self._population_size = int(population_size)
        self._n_full = 3 * self._population_size
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self._reset_state()

    # -- state handling -------------------------------------------------------------

    def _reset_state(self) -> None:
        self._study_name: str | None = None
        self._space: dict[str, BaseDistribution] | None = None
        self._names: list[str] = []
        self._lb = np.empty(0)
        self._ub = np.empty(0)
        self._A = np.empty((0, 0))
        self._fitA = np.empty(0)
        self._best = np.empty(0)
        self._noise = np.empty((0, 0))
        self._j0 = np.empty(0, dtype=int)
        self._generation = -1
        # Vectors (internal representation) to be evaluated in the current generation.
        self._candidates = np.empty((0, 0))
        self._pending: deque[int] = deque()  # Candidate indices not yet given to a trial.
        self._assigned: dict[int, int] = {}  # Trial number -> candidate index.
        self._results: dict[int, tuple[np.ndarray, float]] = {}  # Candidate index -> result.

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def reseed_rng(self) -> None:
        with self._lock:
            self._rng = np.random.default_rng()
            super().reseed_rng()

    # -- search space ---------------------------------------------------------------

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        space = super().infer_relative_search_space(study, trial)
        return {name: dist for name, dist in space.items() if _is_numerical(dist)}

    def _initialize(self, study_name: str, space: dict[str, BaseDistribution]) -> None:
        self._reset_state()
        self._study_name = study_name
        self._space = dict(space)
        self._names = list(space.keys())
        bounds = np.array([_internal_bounds(space[n]) for n in self._names], dtype=float)
        self._lb, self._ub = bounds[:, 0], bounds[:, 1]
        N, D = self._population_size, len(self._names)
        self._A = self._lb + self._rng.random((self._n_full, D)) * (self._ub - self._lb)
        self._noise = np.zeros((N, D))
        self._start_generation(self._A.copy())

    def _start_generation(self, candidates: np.ndarray) -> None:
        self._generation += 1
        self._candidates = candidates
        self._pending = deque(range(len(candidates)))
        self._assigned = {}
        self._results = {}

    # -- MDE generation update ------------------------------------------------------

    def _finish_generation(self) -> None:
        n = len(self._candidates)
        X = np.array([self._results[i][0] for i in range(n)])
        fit = np.array([self._results[i][1] for i in range(n)])
        if self._generation == 0:
            self._A, self._fitA = X, fit
        else:
            j0 = self._j0
            B, fitB = self._A[j0].copy(), self._fitA[j0].copy()
            better = fit < fitB
            B[better], fitB[better] = X[better], fit[better]
            self._A[j0], self._fitA[j0] = B, fitB
            self._noise = _mde_noise(B, self._rng)
        self._best = self._A[int(np.argmin(self._fitA))].copy()

        # Next generation: draw a sub-population and build its trial vectors.
        self._j0 = self._rng.permutation(self._n_full)[: self._population_size]
        B = self._A[self._j0].copy()
        T = _mde_bound(
            _mde_trial(B, self._best, self._noise, self._rng), self._lb, self._ub, self._rng
        )
        self._start_generation(T)

    # -- Optuna hooks -----------------------------------------------------------------

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if len(study.directions) > 1:
            raise ValueError("MDESampler does not support multi-objective optimization.")
        if len(search_space) == 0:
            return {}

        with self._lock:
            if self._study_name != study.study_name or self._space != search_space:
                if self._space is not None:
                    warnings.warn(
                        "The search space (or study) changed; MDESampler restarts with a new "
                        "random population."
                    )
                self._initialize(study.study_name, search_space)

            if not self._pending:
                # All trial vectors of the current generation are being evaluated by other
                # (parallel) workers. Fall back to random sampling for this trial.
                return {}

            idx = self._pending.popleft()
            self._assigned[trial.number] = idx
            x = self._candidates[idx]
            return {
                name: _to_external(search_space[name], float(x[k]))
                for k, name in enumerate(self._names)
            }

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        with self._lock:
            if self._space is None or study.study_name != self._study_name:
                return
            idx = self._assigned.pop(trial.number, None)
            if idx is None:
                return

            # Use the parameters that were actually evaluated (they may differ from the
            # proposal, e.g. for enqueued trials or rounded integer parameters).
            x = self._candidates[idx].copy()
            for k, name in enumerate(self._names):
                if name in trial.params:
                    x[k] = _to_internal(self._space[name], trial.params[name])
            x = np.clip(x, self._lb, self._ub)

            if state == TrialState.COMPLETE and values is not None:
                sign = 1.0 if study.direction == StudyDirection.MINIMIZE else -1.0
                fitness = sign * float(values[0])
            else:
                fitness = math.inf  # Pruned or failed trials never enter the population.
            if math.isnan(fitness):
                fitness = math.inf
            self._results[idx] = (x, fitness)

            if len(self._results) == len(self._candidates):
                self._finish_generation()


__all__ = ["MDESampler"]
