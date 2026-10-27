from __future__ import annotations

from typing import Any
from typing import Literal

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from ._acquisition import maximize_sdp
from ._annealing import maximize_sa
from ._horseshoe import HorseshoeState
from ._horseshoe import sample_horseshoe
from ._model import second_order_features


_logger = optuna.logging.get_logger(__name__)


def _is_binary(distribution: BaseDistribution) -> bool:
    if isinstance(distribution, IntDistribution):
        return (
            distribution.low == 0
            and distribution.high == 1
            and distribution.step == 1
            and not distribution.log
        )
    if isinstance(distribution, CategoricalDistribution):
        return len(distribution.choices) == 2
    return False


def _to_binary(distribution: BaseDistribution, value: Any) -> float:
    # For a categorical parameter, the first choice maps to 0 and the second to 1.
    return float(distribution.to_internal_repr(value))


def _from_binary(distribution: BaseDistribution, bit: float) -> Any:
    return distribution.to_external_repr(float(bit))


class BOCSSampler(BaseSampler):
    """Bayesian Optimization of Combinatorial Structures (Baptista & Poloczek, 2018).

    BOCS models the objective over binary parameters with a second-order polynomial
    whose coefficients have a sparsity-inducing horseshoe prior. In each trial it
    draws one coefficient vector from the posterior (Thompson sampling) and
    proposes the binary point that maximizes it, found either with a semidefinite
    relaxation and randomized rounding (BOCS-SDP) or with simulated annealing (BOCS-SA).

    Binary parameters are declared either with ``trial.suggest_int(name, 0, 1)`` or
    with ``trial.suggest_categorical(name, [a, b])``. Other parameters are sampled by
    ``independent_sampler``.

    Args:
        acquisition:
            ``"sdp"`` (default) maximizes the sampled model with a semidefinite
            relaxation; ``"sa"`` uses simulated annealing, which is cheaper for many
            binary parameters.
        n_startup_trials:
            Number of initial trials sampled by ``independent_sampler``.
        penalty:
            Weight of the regularizer ``penalty * sum(x)`` subtracted from the model
            when proposing a point, in units of the standardized objective. A
            positive value favors points with fewer ones.
        n_burnin:
            Number of Gibbs sweeps before the first posterior sample.
        n_gibbs_steps:
            Number of Gibbs sweeps per trial afterwards. The chain is warm-started
            from the previous trial, so far fewer sweeps are needed than for burn-in.
        n_rounding:
            Number of randomized roundings of the semidefinite relaxation per
            rounding method. Only used with ``acquisition="sdp"``.
        seed:
            Seed for the random number generator.
        independent_sampler:
            Sampler for the startup trials and for non-binary parameters. Defaults
            to :class:`~optuna.samplers.RandomSampler` with the same seed.
        warn_independent_sampling:
            Whether to warn when a non-binary parameter is sampled independently
            after the startup trials.
    """

    def __init__(
        self,
        *,
        acquisition: Literal["sdp", "sa"] = "sdp",
        n_startup_trials: int = 10,
        penalty: float = 0.0,
        n_burnin: int = 1000,
        n_gibbs_steps: int = 100,
        n_rounding: int = 100,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        warn_independent_sampling: bool = True,
    ) -> None:
        if acquisition not in ("sdp", "sa"):
            raise ValueError(f"acquisition must be 'sdp' or 'sa', got {acquisition!r}.")
        if n_burnin < 0 or n_gibbs_steps < 1 or n_rounding < 1:
            raise ValueError("n_burnin must be nonnegative and the other counts positive.")
        self._acquisition = acquisition
        self._n_startup_trials = n_startup_trials
        self._penalty = penalty
        self._n_burnin = n_burnin
        self._n_gibbs_steps = n_gibbs_steps
        self._n_rounding = n_rounding
        self._rng = np.random.default_rng(seed)
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._warn_independent_sampling = warn_independent_sampling
        self._intersection_search_space = optuna.search_space.IntersectionSearchSpace()
        # Gibbs chain state kept between trials, valid for the parameter names below.
        self._state: HorseshoeState | None = None
        self._state_param_names: list[str] = []

    def reseed_rng(self) -> None:
        self._rng = np.random.default_rng()
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {
            name: distribution
            for name, distribution in self._intersection_search_space.calculate(study).items()
            if _is_binary(distribution)
        }

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}
        if len(study.directions) > 1:
            raise ValueError("BOCSSampler does not support multi-objective optimization.")

        names = sorted(search_space)
        trials = [
            t
            for t in study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if all(name in t.params for name in names)
        ]
        if len(trials) < self._n_startup_trials:
            return {}

        x = np.array([[_to_binary(search_space[n], t.params[n]) for n in names] for t in trials])
        values = np.array([t.value for t in trials], dtype=float)
        if study.direction == StudyDirection.MINIMIZE:
            values = -values  # BOCS maximizes.
        std = values.std()
        y = (values - values.mean()) / (std if std > 0 else 1.0)

        if self._state is None or names != self._state_param_names:
            state, n_sweeps = None, self._n_burnin + 1
        else:
            state, n_sweeps = self._state, self._n_gibbs_steps
        samples, self._state = sample_horseshoe(
            second_order_features(x), y, 1, self._rng, n_burnin=n_sweeps - 1, state=state
        )
        self._state_param_names = names

        if self._acquisition == "sdp":
            x_next = maximize_sdp(
                samples[0],
                len(names),
                self._rng,
                penalty=self._penalty,
                n_rounding=self._n_rounding,
            )
        else:
            x_next = maximize_sa(samples[0], len(names), self._rng, penalty=self._penalty)
        return {n: _from_binary(search_space[n], bit) for n, bit in zip(names, x_next)}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if self._warn_independent_sampling and not _is_binary(param_distribution):
            n_complete = len(study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
            if n_complete >= self._n_startup_trials:
                _logger.warning(
                    f"BOCSSampler only models binary parameters; '{param_name}' in trial "
                    f"{trial.number} is sampled by {type(self._independent_sampler).__name__}."
                )
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: list[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)
