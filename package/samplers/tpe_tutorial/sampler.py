from __future__ import annotations

import numpy as np
from optuna.distributions import BaseDistribution
from optuna.samplers import TPESampler
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

from .components import GammaFunc
from .components import WeightFunc
from .parzen_estimator import _CustomizableParzenEstimator
from .parzen_estimator import _CustomizableParzenEstimatorParameters


class CustomizableTPESampler(TPESampler):
    def __init__(
        self,
        consider_prior: bool = True,
        prior_weight: float = 1.0,
        consider_magic_clip: bool = True,
        n_startup_trials: int = 10,
        n_ei_candidates: int = 24,
        seed: int | None = None,
        *,
        categorical_prior_weight: float | None = None,
        multivariate: bool = False,
        warn_independent_sampling: bool = True,
        b_magic_exponent: float = 1.0,
        min_bandwidth_factor: float = 0.01,
        gamma_strategy: str = "linear",
        gamma_beta: float = 0.1,
        weight_strategy: str = "old-decay",
        bandwidth_strategy: str = "hyperopt",
    ):
        gamma = GammaFunc(strategy=gamma_strategy, beta=gamma_beta)
        weights = WeightFunc(strategy=weight_strategy)
        super().__init__(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=True,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            gamma=gamma,
            weights=weights,
            seed=seed,
            multivariate=multivariate,
            warn_independent_sampling=warn_independent_sampling,
        )
        self._parzen_estimator_cls = _CustomizableParzenEstimator
        self._parzen_estimator_parameters = _CustomizableParzenEstimatorParameters(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            weights=weights,
            multivariate=multivariate,
            b_magic_exponent=b_magic_exponent,
            min_bandwidth_factor=min_bandwidth_factor,
            bandwidth_strategy=bandwidth_strategy,
            categorical_prior_weight=categorical_prior_weight,
        )
        self._weight_strategy = weight_strategy

    def _build_parzen_estimator(
        self,
        study: Study,
        search_space: dict[str, BaseDistribution],
        trials: list[FrozenTrial],
        handle_below: bool,
    ) -> _ParzenEstimator:
        if study._is_multi_objective() or self._weight_strategy != "EI" or not handle_below:
            return super()._build_parzen_estimator(study, search_space, trials, handle_below)

        # Not multi-objective and EI and below.
        below_trial_numbers = set([t.number for t in trials])
        sign = 1 if study.direction == StudyDirection.MINIMIZE else -1
        threshold = min(
            sign * t.value
            for t in study._get_trials(
                deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED), use_cache=True
            )
            if t.number not in below_trial_numbers
        )
        if np.isinf(threshold):
            parzen_estimator_parameters = self._parzen_estimator_parameters
            weights_below = np.ones(len(trials))
        else:
            loss_vals = np.asarray([sign * t.value for t in trials])
            weights_below = np.maximum(1e-12, threshold - loss_vals)
            parzen_estimator_parameters = self._parzen_estimator_parameters._replace(
                prior_weight=np.mean(weights_below)
            )

        observations = self._get_internal_repr(trials, search_space)
        return self._parzen_estimator_cls(
            observations, search_space, parzen_estimator_parameters, weights_below
        )
