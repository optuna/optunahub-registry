"""Lévy Flight Random Walk Sampler for Optuna.

This sampler implements a search strategy based on Lévy flights, a type of
random walk where step lengths are drawn from a heavy-tailed (Lévy stable)
distribution. Lévy flights naturally balance exploration and exploitation:
most steps are small (local search), but occasional large jumps escape
local optima. This behavior is observed in the foraging patterns of many
animals and is used in nature-inspired algorithms like Cuckoo Search.

Reference:
    Yang, X.-S. & Deb, S. (2010). Engineering Optimisation by Cuckoo Search.
    International Journal of Mathematical Modelling and Numerical Optimisation,
    1(4), 330–343. https://doi.org/10.1504/IJMMNO.2010.035430
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import RandomSampler


class LevyFlightSampler(optuna.samplers.BaseSampler):
    """Sampler based on Lévy flight random walks.

    This sampler maintains a *current best* position and proposes new
    candidates by taking a Lévy-flight step from that position. The step
    length is drawn from a Lévy stable distribution approximated via the
    Mantegna algorithm, which is accurate and efficient.

    The algorithm naturally transitions from wide exploration (early trials,
    large effective steps) toward finer exploitation as the run progresses,
    controlled by ``step_scale``.

    Compared to pure :class:`~optuna.samplers.RandomSampler`, Lévy flights
    converge faster on unimodal functions while still being capable of
    escaping shallow local optima — at the cost of being stateful (the
    sampler tracks the current best across trials).

    Args:
        beta (float):
            Stability index of the Lévy distribution.  Must be in ``(0, 2]``.
            ``beta=2`` recovers a Gaussian random walk; ``beta=1`` gives a
            Cauchy distribution (very heavy tails). A value around ``1.5`` is
            recommended for most optimisation problems.
        step_scale (float):
            Global scaling factor applied to every Lévy step.  Smaller values
            concentrate search near the current best; larger values allow
            wider jumps.  Defaults to ``0.1`` (10 % of the search range).
        seed (int | None):
            Seed for the internal random number generator.  Use an integer
            for reproducible runs.

    Example:
        .. code-block:: python

            import optuna
            import optunahub

            def objective(trial):
                x = trial.suggest_float("x", -5, 5)
                y = trial.suggest_float("y", -5, 5)
                return (x - 1.5) ** 2 + (y + 2.0) ** 2

            mod = optunahub.load_module("samplers/levy_flight_sampler")
            sampler = mod.LevyFlightSampler(beta=1.5, step_scale=0.1, seed=42)
            study = optuna.create_study(sampler=sampler)
            study.optimize(objective, n_trials=100)
            print(study.best_params, study.best_value)
    """

    def __init__(
        self,
        beta: float = 1.5,
        step_scale: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if not (0.0 < beta <= 2.0):
            raise ValueError(f"`beta` must be in (0, 2], got {beta}.")
        if step_scale <= 0:
            raise ValueError(f"`step_scale` must be positive, got {step_scale}.")

        self._beta = beta
        self._step_scale = step_scale
        self._rng = np.random.RandomState(seed)
        self._random_sampler = RandomSampler(seed=seed)

        # Pre-compute Mantegna sigma (depends only on beta, so computed once).
        self._sigma = self._mantegna_sigma(beta)

    # ------------------------------------------------------------------
    # Public Optuna sampler interface
    # ------------------------------------------------------------------

    def infer_relative_search_space(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> dict[str, BaseDistribution]:
        return optuna.search_space.intersection_search_space(study.get_trials(deepcopy=False))

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        if not search_space:
            return {}

        completed = study.get_trials(states=[optuna.trial.TrialState.COMPLETE], deepcopy=False)

        # Not enough history — fall back to independent (random) sampling.
        if len(completed) < 2:
            return {}

        # Find the current best trial.
        if study.direction == optuna.study.StudyDirection.MINIMIZE:
            best_trial = min(completed, key=lambda t: t.value)  # type: ignore[arg-type]
        else:
            best_trial = max(completed, key=lambda t: t.value)  # type: ignore[arg-type]

        params: dict[str, Any] = {}
        for name, dist in search_space.items():
            if name not in best_trial.params:
                # Parameter not present in best trial — skip (handled by independent).
                continue

            # Handle categorical first — no float cast possible.
            if isinstance(dist, CategoricalDistribution):
                params[name] = self._rng.choice(dist.choices)
                continue

            current = float(best_trial.params[name])

            if isinstance(dist, FloatDistribution):
                low, high = dist.low, dist.high
                step = self._levy_step(high - low)
                new_val = current + step

                if dist.log:
                    # Work in log-space for log-uniform distributions.
                    log_low, log_high = math.log(dist.low), math.log(dist.high)
                    log_current = math.log(max(current, dist.low))
                    log_step = self._levy_step(log_high - log_low)
                    new_val = math.exp(np.clip(log_current + log_step, log_low, log_high))
                else:
                    new_val = float(np.clip(new_val, low, high))

                if dist.step is not None:
                    # Round to the nearest grid point.
                    n_steps = round((new_val - low) / dist.step)
                    new_val = float(np.clip(low + n_steps * dist.step, low, high))

                params[name] = new_val

            elif isinstance(dist, IntDistribution):
                low, high = dist.low, dist.high
                step = self._levy_step(float(high - low))
                new_val = int(round(np.clip(current + step, low, high)))

                if dist.step != 1:
                    # Snap to the nearest valid integer step.
                    n_steps = round((new_val - low) / dist.step)
                    new_val = int(np.clip(low + n_steps * dist.step, low, high))

                params[name] = new_val

            else:
                # Unknown distribution type — skip; handled by independent sampler.
                continue

        return params

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._random_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _levy_step(self, search_range: float) -> float:
        """Draw a single Lévy-flight step scaled to ``search_range``.

        Uses the Mantegna algorithm to approximate a Lévy stable variate
        efficiently without requiring special-function libraries.
        """
        # Mantegna algorithm: u ~ N(0, sigma^2), v ~ N(0, 1)
        u = self._rng.normal(0.0, self._sigma)
        v = self._rng.normal(0.0, 1.0)
        step = u / (abs(v) ** (1.0 / self._beta))
        return float(self._step_scale * search_range * step)

    @staticmethod
    def _mantegna_sigma(beta: float) -> float:
        """Compute the Mantegna sigma parameter for the given stability index.

        sigma = ( Gamma(1+beta)*sin(pi*beta/2) /
                  (Gamma((1+beta)/2) * beta * 2^((beta-1)/2)) )^(1/beta)

        This is numerically stable for beta in (0, 2].
        """
        num = math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0)
        den = math.gamma((1.0 + beta) / 2.0) * beta * (2.0 ** ((beta - 1.0) / 2.0))
        return (num / den) ** (1.0 / beta)